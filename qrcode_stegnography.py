import qrcode
import cv2, os
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode

def to_bin(data):
    """Convert `data` to binary format as string"""
    if isinstance(data, str):
        return ''.join([ format(ord(i), "08b") for i in data ])
    elif isinstance(data, bytes):
        return ''.join([ format(i, "08b") for i in data ])
    elif isinstance(data, np.ndarray):
        return [ format(i, "08b") for i in data ]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    elif isinstance(data, tuple): #[ format(i, "08b") for i in data ]
        return ''.join([ format(i, "08b") for i in data ])
    else:
        raise TypeError("Type not supported.")

def image_to_excel(image, name, wt="w"):
    print(f"Converting {name} to excel...")
    pwd = os.getcwd()
    path = pwd + f"/{name}.xlsx"
    if isinstance(image, np.ndarray):
        ls = image.tolist()
        df = pd.DataFrame(ls)
    elif isinstance(image, list):
        df = pd.DataFrame(image)
        
    if wt == "a":
        excel_data_df = pd.read_excel(path, sheet_name="image", header=None, index_col=None)
        null = [np.nan for none in range(0, len(excel_data_df.columns))]
        df_null = pd.DataFrame([null,null])
        df = pd.concat([excel_data_df, df_null, df], ignore_index=True)
    with pd.ExcelWriter(path) as writer:
        print("Writing to excel...")
        df.to_excel(writer, sheet_name="image", index=False, header=False) 
    
def encode(image, data, qr_name="qrcode.png", qr_save=False):
    # read the image
    image = cv2.imread(image)
    #image_to_excel(image)
    height, width, channels = image.shape
    qr = qrcode.QRCode(version=1,
                   error_correction=qrcode.constants.ERROR_CORRECT_L,
                   box_size=3,
                   border=0,
                   )
    qr.add_data(data=data)
    qr.make(fit=True)
    image_qr = qr.make_image(fill_color="black", back_color="white")
    height_qr, width_qr = image_qr.size
    width_tm = int(width_qr / channels)
    if height_qr > height or width_tm > width:
        print("[!] Insufficient image size, need bigger image or less data.", height_qr, width_tm, height, width)
        exit(0)
    if qr_save:
        print("[+] Saving QR code...")
        image_qr.save(qr_name)
    print(image_qr.size)
    encode_data = np.asarray(image_qr)
    encode_shape_data = to_bin(encode_data.shape)
    encode_shape_data_len = len(encode_shape_data)
    print("[+] Encoding data...")
    # image_to_excel(image, "image" )
    # image_to_excel(encode_data, "encode_data_qr")
    # shape data encoding
    shape_index = 0
    for pixel in image[0]:
        r, g, b = to_bin(pixel)
        if shape_index < encode_shape_data_len:
            pixel[0] = int(r[:-1] + encode_shape_data[shape_index], 2)
            shape_index += 1
        if shape_index < encode_shape_data_len:
            pixel[1] = int(g[:-1] + encode_shape_data[shape_index], 2)
            shape_index += 1
        if shape_index < encode_shape_data_len:
            pixel[2] = int(b[:-1] + encode_shape_data[shape_index], 2)
            shape_index += 1
        if shape_index == encode_shape_data_len:
            break
    # encode the data into image
    qr_r_index = 0
    qr_c_index = 0
    
    for row in image[1:height_qr]:
        for pixel in row[:width_tm]:
            r, g, b = to_bin(pixel)
            if qr_r_index < height_qr:
                status_r = encode_data[qr_r_index][qr_c_index]
                status_g = encode_data[qr_r_index][qr_c_index + 1]
                status_b = encode_data[qr_r_index][qr_c_index + 2]
                if status_r == False:
                    pixel[0] = int(r[:-1] + "0", 2)
                if status_r == True:
                    pixel[0] = int(r[:-1] + "1", 2)
                if status_g == False:
                    pixel[1] = int(g[:-1] + "0", 2)
                if status_g == True:
                    pixel[1] = int(g[:-1] + "1", 2)
                if status_b == False:
                    pixel[2] = int(b[:-1] + "0", 2)
                if status_b == True:
                    pixel[2] = int(b[:-1] + "1", 2)
                qr_c_index += 3
                if qr_c_index == width_qr:
                    qr_r_index += 1
                    qr_c_index = 0    
                
    #image_to_excel(image, "encode_image")
    return image
    
def image_decode(output_image, qr_name="qrcode.png", qr_save=False):
    print("[+] Decoding data...")
    image = cv2.imread(output_image)
    #image_to_excel(image, "decode_image")
    
    # shape data decoding
    binary_data_shape = ""
    for pixel in image[0]:
        r, g, b = to_bin(pixel)
        binary_data_shape += r[-1] # 0000000[0]
        binary_data_shape += g[-1] # 0000000[0]
        binary_data_shape += b[-1] # 0000000[0]
    shape_data = [ binary_data_shape[i: i+8] for i in range(0, len(binary_data_shape), 8) ]
    shape_qr = [ int(i, 2) for i in shape_data ][:2]
    print(shape_qr)
    if (shape_qr[0] == 0 or shape_qr[1] == 0) or shape_qr[0] != shape_qr[1]:
        print("Could it have been hacked?")
        exit(0)
        
    shape_w_h = shape_qr[0]#, shape_qr[1]
    shape_w = int(shape_w_h / 3)
    # print("[+] QR code shape:", shape_w)
    # decode the qrcode from image
    qr_r_index = 0
    qr_c_index = 0
    binary_qr = []
    for row in image[1:shape_w_h+1]:
        binary_qr_row = []
        for pixel in row[:shape_w]:
            r, g, b = to_bin(pixel)
            status_r = int(r[-1])
            status_g = int(g[-1])
            status_b = int(b[-1])
            if status_r == 0:
                binary_qr_row.append(0)
            if status_r == 1:
                binary_qr_row.append(255)
            if status_g == 0:
                binary_qr_row.append(0)
            if status_g == 1:
                binary_qr_row.append(255)
            if status_b == 0:
                binary_qr_row.append(0)
            if status_b == 1:
                binary_qr_row.append(255)
            qr_c_index += 3
            if qr_c_index == shape_w_h:
                #qr_r_index += 1
                qr_c_index = 0
                binary_qr.append(binary_qr_row)
                break
    binary_qr = np.array(binary_qr)
    #image_to_excel(binary_qr, "decode_data_qr")
    decoded_data = decode(binary_qr)[0]
    if qr_save:
        print("[+] Saving QR code...")
        cv2.imwrite(qr_name, binary_qr)
    #image_to_excel(binary_qr, "decode_data_qr")
    return decoded_data.data.decode("utf-8")
"""
"""

def ssim(image, output_image):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    image = image.astype(np.float64)
    output_image = output_image.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(image, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(output_image, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    singal1_sq = cv2.filter2D(image ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    singal2_sq = cv2.filter2D(output_image ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    singal1_2 = cv2.filter2D(image * output_image, -1, window)[5:-5, 5:-5] - mu1 * mu2
    return ((2 * mu1 * mu2 + c1) * (2 * singal1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (singal1_sq + singal2_sq + c2)).mean()

def calculate_ssim(image, output_image):
    if not image.shape == output_image.shape:
        raise ValueError("Input Imagees must have the same dimensions...")
    if image.ndim == 2:
        return ssim(image, output_image)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(image, output_image))
            return np.array(ssims).mean()
        elif image.shape[2] == 1:
            return ssim(np.squeeze(image), np.squeeze(output_image))
    else:
        raise ValueError("Wrong input image dimensions...")

def calculate(image, output_image):
    import math
    result = []
    # read the image
    image = cv2.imread(image)
    output_image = cv2.imread(output_image)
    """ Calculate the MSE (Mean Squared Error) of two images. The MSE is the sum of the """
    """ Calculate the PSNR (Peak Signal to Noise Ratio) of two images. The PSNR is the """
    mse = np.mean((image - output_image) ** 2)
    if mse == 0:
        result.append(0)
        result.append(0)
        result.append(100)
    else:
        result.append(mse)
        result.append(math.sqrt(mse))
        result.append(20 * math.log10(255.0 / math.sqrt(mse)))
        
    """ Calculate the SSIM (Structural Similarity Index) of two images. The SSIM is the """
    ssim = calculate_ssim(image, output_image)
    result.append(ssim)
    
    print("[+] Calculating...")
    print("[+] MSE: ", round(result[0],5))
    print("[+] RMSE: ", round(result[1],5))
    print("[+] PSNR: ", round(result[2],5))
    print("[+] SSIM: ", round(result[3],5))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Steganography encoder/decoder, this Python scripts encode data within images.")
    parser.add_argument("-t", "--text", help="The text data to encode into the image, this only should be specified for encoding")
    parser.add_argument("-f", "--file", help="The file to hide into the image, this only should be specified while encoding")
    parser.add_argument("-e", "--encode", help="Encode the following image")
    parser.add_argument("-d", "--decode", help="Decode the following image")
    parser.add_argument("-q", "--qrcode", help="Creating a Qr Code file", action=argparse.BooleanOptionalAction)
    
    # parse the arguments
    args = parser.parse_args()
    if args.encode:
        # if the encode argument is specified
        if args.text:
            secret_data = args.text
        elif args.file:
            with open(args.file, "rb") as f:
                secret_data = f.read()
        image = args.encode
        # split the absolute path the file
        path, file = os.path.split(image)
        # split the filename and the image extension
        filename, ext = os.path.splitext(file)            
        output_image = os.path.join(path, filename + "_encoded" + ext)
        if args.qrcode:
            qrcode_name = os.path.join(path, filename + "_qrcode.png")
            endoded_image = encode(image, secret_data, qr_name=qrcode_name, qr_save=True)
        else:
            endoded_image = encode(image, secret_data, qr_save=False)
        # save the encoded image
        cv2.imwrite(output_image, endoded_image)
        print("[+] Saved encoded image to:", output_image)
        calculate(image, output_image)            
    if args.decode:
        # if the decode argument is specified
        image = args.decode
        if args.file:
            if args.qrcode:
                decoded_data = image_decode(image, qr_save=True)
            else:
                decoded_data = image_decode(image, qr_save=False)
            with open(args.file, "wb") as f:
                f.write(decoded_data)
            print(f"[+] Saved decoded data to: {args.file}")
        else:
            if args.qrcode:
                decoded_data = image_decode(image, qr_save=True)
            else:
                decoded_data = image_decode(image, qr_save=False)
            print("[+] Decoded data:", decoded_data)
# Encoding
# clear && py qrcode_stegnography2.py -e mandrill.png -t "vedat önal merhaba"
# clear && py qrcode_stegnography2.py -e mandrill.png -t "vedat önal merhaba" -q
# Decoding
# clear && py qrcode_stegnography2.py -d mandrill_encoded.png
# clear && py qrcode_stegnography2.py -d mandrill_encoded.png -q
