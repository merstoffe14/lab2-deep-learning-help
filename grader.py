import torch.nn.functional

import tensors
from options.options import Options
from utilities.utils import mse, not_implemented


def test_mse():
    test_input = torch.randn([3, 50, 100])
    test_target = torch.randn([3, 50, 100])
    error = mse(test_input, test_target)
    if error == not_implemented():
        print("MSE not yet implemented... (0/1)")
    elif error.shape != torch.Size([]):
        print("MSE not correctly implemented... (0/1)")
        print("The loss shape is incorrect.")
        print(f"Yours: {error.shape}")
        print(f"Correct: {torch.Size([])}")
    else:
        error2 = torch.nn.functional.mse_loss(test_input, test_target)
        if error.item() == error2.item():
            print("MSE implemented correctly! (1/1)")
        else:
            print("MSE not correctly implemented... (0/1)")
            print("Hint: did you forget to take the mean?")


def test_image_matrix():
    created_image = tensors.create_image(options).to(options.device)
    if created_image == not_implemented():
        print("create_target_image not yet implemented... (0/1)")
    else:
        rgb = [[[0.5021, 0.2843, 0.1935],
                [0.8017, 0.5914, 0.7038]],
               [[0.1138, 0.0684, 0.5483],
                [0.8733, 0.6004, 0.5983]],
               [[0.9047, 0.6829, 0.3117],
                [0.6258, 0.2893, 0.9914]]]
        good_image = torch.FloatTensor(rgb).to(options.device)
        if torch.equal(good_image, created_image):
            print("create_target_image implemented correctly! (1/1)")
        else:
            print("create_target_image not implemented correctly... (0/1)")
            print(f"Your image shape: {created_image.shape}\nOur image shape: {good_image.shape}")
            print(f"If the shapes match, perhaps there is an error in the pixel values...")


def test_tensor_forward():
    input_tensor = torch.FloatTensor([0.2, 0.1, 0.5, 0.9]).to(options.device)
    weights = torch.FloatTensor([0.4, -0.1, 0.4, -0.5]).to(options.device)
    output = tensors.tensor_forward(weights, input_tensor)
    if output == not_implemented():
        print("tensor_forward not yet implemented... (0/1)")
    elif output.shape != torch.Size([]):
        print("Something is wrong with the output shape... (0/1)")
        print(f"Yours: {output.shape}")
        print(f"Target: {torch.Size([])}")
    else:
        if output.item() - (-0.18) < 0.000001:
            print("tensor_forward implemented correctly! (1/1)")
        else:
            print("tensor_forward not implemented correctly... (0/1)")


if __name__ == "__main__":
    options = Options()
    print("\nQUESTION 1:\n")
    test_mse()
    print("\nQUESTION 2:\n")
    test_image_matrix()
    print("\nQUESTION 3:\n")
    test_tensor_forward()
