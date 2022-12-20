from PIL import Image
import numpy as np

image = Image.open("./files/Q6_pic.jpg")
pix = np.array(image)
print(pix.shape)
red = pix[:, :, 0].astype('uint8')
blue = pix[:, :, 1].astype('uint8')
green = pix[:, :, 2].astype('uint8')
U1, S1, V1 = np.linalg.svd(red)
U2, S2, V2 = np.linalg.svd(blue)
U3, S3, V3 = np.linalg.svd(green)


def calcLowRank(k, U, S, V):
    S_diag = np.diag(S)
    Uk = U[:, 0:k]
    Sk = S_diag[0:k, 0:k]
    #VkT = np.transpose(V[:, 0:k])
    Vk = V[0:k, :]
    result = np.matmul(Sk, Vk)
    return np.matmul(Uk, result)

k=200
redK = calcLowRank(k, U1, S1, V1)
blueK = calcLowRank(k, U2, S2, V2)
greenK = calcLowRank(k, U3, S3, V3)
new_image = np.zeros(pix.shape)
print(new_image.shape)
new_image[:, :, 0] = redK
new_image[:, :, 1] = blueK
new_image[:, :, 2] = greenK

finalImage = Image.fromarray(new_image.astype('uint8'))
finalImage.show()


