import math

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    MAX = 255.0
    return 20 * math.log10(MAX / math.sqrt(mse))