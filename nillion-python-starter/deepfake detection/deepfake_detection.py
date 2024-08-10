


import torch
from PIL import Image
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr
import numpy as np
import asyncio
import os
from dotenv import load_dotenv
import pprint
import sys
from nada_dsl import *

import py_nillion_client as nillion
from utils.func import compute_prediction, compute_scaled_data, calc_scaling_factor

from config import (
    CONFIG_PROGRAM_NAME,
    CONFIG_TEST_PARTY_1,
    CONFIG_HP_PARTIES,
    CONFIG_NUM_PARAMS
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers.nillion_client_helper import create_nillion_client
from helpers.nillion_keypath_helper import getUserKeyFromFile, getNodeKeyFromFile

load_dotenv()


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

try:
    checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading checkpoint: {e}")

def predict(image, true_label):
    try:
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        face = mtcnn(pil_image)
        if face is None:
            return "No face detected", None

        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        face = face.to(DEVICE, dtype=torch.float32) / 255.0

        target_layers = [model.block8.branch1[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8'), 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            
            real_prediction = 1 - output.item()
            fake_prediction = output.item()
            
            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }

            final_prediction = "real" if real_prediction > fake_prediction else "fake"
        
        result = f"Confidences: Real - {confidences['real']:.4f}, Fake - {confidences['fake']:.4f}\n"
        result += f"True Label: {true_label}\n"
        result += f"Final Prediction: {final_prediction}"

        # Nillion Blind Compute Integration
        asyncio.run(nillion_blind_compute(face, confidences, final_prediction, true_label))

        return result, face_with_mask

    except Exception as e:
        return str(e), None

async def nillion_blind_compute(face, confidences, final_prediction, true_label):
    load_dotenv()

    print("\n\n******* Nillion Blind Compute Integration *******\n\n")
    
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    program_mir_path = f"../programs-compiled/{CONFIG_PROGRAM_NAME}.nada.bin"
    num_params = CONFIG_NUM_PARAMS

    # Setup Nillion Client for test patient
    print("\nSetting up test patient client...\n")
    try:
        client_test_patient = create_nillion_client(
            getUserKeyFromFile(CONFIG_TEST_PARTY_1["userkey_file"]),
            getNodeKeyFromFile(CONFIG_TEST_PARTY_1["nodekey_file"]),
        )
    except Exception as e:
        print(f"Error creating Nillion client for test patient: {e}")
        return

    party_id_test_patient = client_test_patient.party_id()
    user_id_test_patient = client_test_patient.user_id()
    print("Party ID Test Patient:", party_id_test_patient)
    print("User ID Test Patient:", user_id_test_patient)

    # Client test patient stores program
    print("\nClient Test Patient Storing program...\n")
    try:
        action_id = await client_test_patient.store_program(
            cluster_id, CONFIG_PROGRAM_NAME, program_mir_path
        )
    except Exception as e:
        print(f"Error storing program for test patient: {e}")
        return

    program_id = f"{user_id_test_patient}/{CONFIG_PROGRAM_NAME}"
    print("\nStored program. action_id:", action_id)
    print("\nStored program_id:", program_id)

    # Create secrets for test patient
    print("\nSetting up secrets for test patient...\n")
    party_test_patient_dict = {}
    scaled_face_data = face.cpu().detach().numpy().flatten()

    for i in range(num_params):
        print(f"face_param_{i}: {scaled_face_data[i]}")
        party_test_patient_dict[f"face_param_{i}"] = nillion.SecretInteger(
            scaled_face_data[i]
        )

    print("\nParty Test Patient:")
    pprint.PrettyPrinter(indent=4).pprint(party_test_patient_dict)

    # Test Patient store secrets
    test_patient_secrets = nillion.Secrets(party_test_patient_dict)

    # Create test patient input bindings for program
    print("\nSetting up test patient input bindings...\n")
    program_bindings = nillion.ProgramBindings(program_id)
    program_bindings.add_input_party(
        CONFIG_TEST_PARTY_1["party_name"], party_id_test_patient
    )

    # Store secrets on the network
    print("\nStoring Test Patients secrets on the network...\n")
    try:
        store_id_test_patient = await client_test_patient.store_secrets(
            cluster_id, program_bindings, test_patient_secrets, None
        )
    except Exception as e:
        print(f"Error storing secrets for test patient: {e}")
        return

    print(f"\nSecrets for Test Patient: {test_patient_secrets} at program_id: {program_id}")
    print(f"\nStore_id: {store_id_test_patient}")

    ###### Setup Health Provider Party clients and store secrets ######

    store_ids = []
    party_ids = []

    for party_info in CONFIG_HP_PARTIES:
        print(f"\nSetting up {party_info['party_name']} client...\n")
        try:
            client_n = create_nillion_client(
                getUserKeyFromFile(party_info["userkey_file"]),
                getNodeKeyFromFile(party_info["nodekey_file"]),
            )
        except Exception as e:
            print(f"Error creating Nillion client for {party_info['party_name']}: {e}")
            continue

        party_id_n = client_n.party_id()
        user_id_n = client_n.user_id()
        party_name = party_info["party_name"]
        print(f"Party ID for {party_info['party_name']}:", party_id_n)
        print(f"User ID for {party_info['party_name']}:", user_id_n)

        # Create secrets for Health Provider parties
        party_n_dict = {}
        dataset = []

        # Add dataset secret based on config
        if party_info["dataset"] == "face_data":
            dataset = scaled_face_data
            party_n_dict["face_weight"] = nillion.SecretInteger(1)
        else:
            print("Error: Invalid dataset")
            return

        # Add secrets to party
        print(f"\nAdding {num_params} data points to {party_name}:")

        for i in range(num_params):
            party_n_dict[f"{party_info['secret_name']}{i}"] = nillion.SecretInteger(dataset[i])
            print(f"{party_info['secret_name']}{i}: {dataset[i]}")

        party_secret = nillion.Secrets(party_n_dict)

        # Create input bindings for the program
        print(f"\nSetting up input bindings for {party_name}...\n")
        secret_bindings = nillion.ProgramBindings(program_id)
        secret_bindings.add_input_party(party_name, party_id_n)

        # Create permissions object
        secrets_permissions = nillion.Permissions()
        secrets_permissions.add_party_write(party_id_test_patient)

        print(f"\nStoring secrets for {party_name}...\n")
        try:
            store_id = await client_n.store_secrets(
                cluster_id, secret_bindings, party_secret, secrets_permissions
            )
        except Exception as e:
            print(f"Error storing secrets for {party_name}: {e}")
            continue

        print(f"\nStored Secrets for {party_name}: {party_secret}")
        print(f"\nStore_id: {store_id}")

        store_ids.append(store_id)
        party_ids.append(party_id_n)

    party_ids_to_store_ids = {party_ids[i]: store_ids[i] for i in range(len(party_ids))}
    print(f"\nparty_ids_to_store_ids: {party_ids_to_store_ids}")

    ###### Start the computation ######

    # All set! Trigger compute program
    print("\nTriggering computation...\n")
    try:
        result = await client_test_patient.compute_program(
            cluster_id, program_bindings, party_ids_to_store_ids, None
        )
        print("\nComputation Result:")
        pprint.PrettyPrinter(indent=4).pprint(result)
    except Exception as e:
        print(f"Error triggering computation: {e}")


def nada_main():
    # Define the parties
    party_test_patient = Party(name="TestPatient")
    party_hp1 = Party(name="HealthProvider1")
    party_hp2 = Party(name="HealthProvider2")

    # Define the input secrets
    face_params = [SecretInteger(Input(name=f"face_param_{i}", party=party_test_patient)) for i in range(CONFIG_NUM_PARAMS)]

    # Define a computation using only nada_dsl types
    # For example, computing the weighted sum of face parameters (adjust based on actual computation)
    weight = SecretInteger(Input(name="weight", party=party_hp1))

    # Initialize weighted_sum as SecretInteger
    weighted_sum = SecretInteger(0)

    for param in face_params:
        # Each operation should be between SecretInteger instances
        weighted_sum = weighted_sum + (param * weight)

    # Define the output
    output = Output(weighted_sum, "computed_output", party_test_patient)

    return [output]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="True Label")
    ],
    outputs=[
        gr.Textbox(label="Prediction Results"),
        gr.Image(label="Face with Heatmap")
    ],
    title="DeepFake Detection",
    description="Upload an image to detect if it's a deepfake."
)

if __name__ == "__main__":
    iface.launch()
