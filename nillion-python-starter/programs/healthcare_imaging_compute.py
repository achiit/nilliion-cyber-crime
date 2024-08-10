from nada_dsl import *


def nada_main():
    num_params = 10  # Number of features we're using for the prediction

    # Define the parties
    party1 = Party(name="Party1")  # Represents the user uploading the image
    party2 = Party(name="Party2")  # Represents the first health provider
    party3 = Party(name="Party3")  # Represents the second health provider

    # Define the secret inputs
    dataset1_weight = SecretInteger(Input(name="dataset1_weight", party=party2))
    dataset2_weight = SecretInteger(Input(name="dataset2_weight", party=party3))
    
    image_features = []  # Features from the user's image
    theta_party2 = []  # Stored parameters from party2
    theta_party3 = []  # Stored parameters from party3

    for i in range(num_params):
        image_features.append(
            SecretInteger(Input(name=f"feature_{i}", party=party1))
        )
        theta_party2.append(
            SecretInteger(Input(name=f"theta_party2_{i}", party=party2))
        )
        theta_party3.append(
            SecretInteger(Input(name=f"theta_party3_{i}", party=party3))
        )

    # Compute the weighted average of the parameters (theta values)
    combined_theta = []
    for theta2, theta3 in zip(theta_party2, theta_party3):
        combined_theta.append(
            ((theta2 * dataset1_weight) / Integer(100))
            + ((theta3 * dataset2_weight) / Integer(100))
        )

    # Compute the prediction by performing element-wise multiplication and summation
    prediction = Integer(0)
    for feature, theta in zip(image_features, combined_theta):
        prediction += feature * theta

    # Output the prediction
    return [
        Output(prediction, "prediction", party1),
    ]
