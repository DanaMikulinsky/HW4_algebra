from PIL import Image
import numpy as np

image = Image.open("q6_files/Q6_pic.jpg")
pix = np.array(image)
print(pix.shape)
red = pix[:, :, 0].astype('uint8')
blue = pix[:, :, 1].astype('uint8')
green = pix[:, :, 2].astype('uint8')
U1, S1, V1 = np.linalg.svd(red)
U2, S2, V2 = np.linalg.svd(blue)
U3, S3, V3 = np.linalg.svd(green)

red_rank = np.count_nonzero(S1)
blue_rank = np.count_nonzero(S2)
green_rank = np.count_nonzero(S3)


def calcLowRank(k, U, S, V):
    S_diag = np.diag(S)
    Uk = U[:, 0:k]
    Sk = S_diag[0:k, 0:k]
    Vk = V[0:k, :]
    result = np.matmul(Sk, Vk)
    return np.matmul(Uk, result)



k_vals=[260, 200, 120, 100, 70, 50, 30, 10, 2]
errors = np.zeros(len(k_vals))
sSquare = pow(np.array(S1), 2)
for i in range(len(k_vals)):
    redK = calcLowRank(k_vals[i], U1, S1, V1)
    blueK = calcLowRank(k_vals[i], U2, S2, V2)
    greenK = calcLowRank(k_vals[i], U3, S3, V3)
    new_image = np.zeros(pix.shape)
    new_image[:, :, 0] = redK
    new_image[:, :, 1] = blueK
    new_image[:, :, 2] = greenK
    finalImage = Image.fromarray(new_image.astype('uint8'))
    errors[i] = sum(sSquare[k_vals[i]+1:])/sum(sSquare)
    finalImage.save("pinkOnWednesday_" + str(k_vals[i]) + "_" + str("{:.5f}".format(errors[i])) + ".png")

k_min = 50
proportions = [k_min/red_rank, k_min/green_rank, k_min/blue_rank]
print("proportions:")
print(proportions)
print("errors:")
print(errors)
