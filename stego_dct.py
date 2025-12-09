import warnings
import re
import cv2
import numpy as np
import pandas

class SteganografiDCT:
    def __init__(self):
        self.Q_TABLE = Q_TABLE = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)

        warnings.filterwarnings('ignore')

    def decode(self, gambar_stego):
        merged_ycrcb = cv2.cvtColor(gambar_stego, cv2.COLOR_RGB2YCrCb)

        blocks_stego, shape_stego, info_stego, blocks_shape_stego = self.pecahan_gambar(merged_ycrcb)
        splited_stego = self.pecah_channel(blocks_stego)
        splited_stego = self.pecah_channel(blocks_stego)
        dct_stego = self.dct(splited_stego)
        quantized_stego = self.quantize_blocks(dct_stego, self.Q_TABLE)
        return self.extract_bits(quantized_stego)


    def extract_bits(self, quantized_blocks):
        
        location = ((0, 1), (1, 0), (1, 1))

        bit_rahasia = '' 
        length_check = 0
        max_bits = 0
        length_tanda = 0

        for block in quantized_blocks:
            for r, c in location:  
                bit1 = str(self.get_lsb_safe(block[1][r][c]))
                bit2 = str(self.get_lsb_safe(block[2][r][c]))
                bit_rahasia += bit1
                bit_rahasia += bit2
            

        hasil_bit_pesan = self.bits_to_text(bit_rahasia)

        match = re.search('p:\d+:p', hasil_bit_pesan)
        bit_rahasia

        if not match:
            return "Tidak ada pesan tersembunyi"
        else:
            max_length = int(match.group().replace('p:', '').replace(':p', ''))
            return hasil_bit_pesan[len(str(max_length)) + 4:len(str(max_length)) + 4 + max_length]



    def encode(self, pesan_rahasia, gambar):
        cover_ycrcb = cv2.cvtColor(gambar, cv2.COLOR_RGB2YCrCb)
        bit_pesan = self.convert_text_bit(pesan_rahasia)
        blocks, shape, info, blocks_shape = self.pecahan_gambar(cover_ycrcb)
        blocks_splited = self.pecah_channel(blocks)
        dct_blocks = self.dct(blocks_splited)
        quantized_blocks = self.quantize_blocks(dct_blocks, self.Q_TABLE)
        stego_blocks = self.embed_bits(quantized_blocks, bit_pesan)
        idct_blocks = self.dequantize_and_idct(stego_blocks)
        stego_gambar = self.merge_blocks(idct_blocks, shape, info, blocks_shape)
        return stego_gambar

    
    def embed_bits(self, quantized_blocks, bit_pesan):
        location = ((0, 1), (1, 0), (1, 1))

        stego_blocks = quantized_blocks

        bit_index = 0 

        for block in stego_blocks:
            for r, c in location:
                
                if bit_index >= len(bit_pesan):
                    break   
                
                block[1][r][c] = self.set_lsb_safe(block[1][r][c], int(bit_pesan[bit_index]))
                bit_index += 1

                if bit_index >= len(bit_pesan):
                    break

                block[2][r][c] = self.set_lsb_safe(block[2][r][c], int(bit_pesan[bit_index]))
                bit_index += 1

            if bit_index >= len(bit_pesan):
                break

        return stego_blocks

    def dequantize_and_idct(self, stego_blocks):
        dequantized_blocks = self.dequantize_blocks(stego_blocks, self.Q_TABLE)
        idct_blocks = self.idct(dequantized_blocks)
        return idct_blocks


    def convert_text_bit(self, text):
        bits = ''.join(format(ord(char), '08b') for char in text)
        new_text = f'p:{str(len(text))}:p{text}'
        return ''.join(format(ord(char), '08b') for char in new_text)
    
    def bits_to_text(self, bits):
        chars = []
        for i in range(0, len(bits), 8): 
            byte = bits[i:i+8]
            if len(byte) < 8:
                continue
            ascii_val = int(byte, 2)  
            chars.append(chr(ascii_val))  
        return ''.join(chars)

    def padding(self, gambar, ukuran_block=8):
        Y, Cr, Cb = cv2.split(gambar) 
        H, W = gambar.shape[:2]

        pad_H = (ukuran_block - H % ukuran_block) % ukuran_block
        pad_W = (ukuran_block - W % ukuran_block) % ukuran_block

        Y  = np.pad(Y,  ((0, pad_H), (0, pad_W)), mode='edge')
        Cr = np.pad(Cr, ((0, pad_H), (0, pad_W)), mode='edge')
        Cb = np.pad(Cb, ((0, pad_H), (0, pad_W)), mode='edge')

        return cv2.merge([Y, Cr, Cb]), (pad_H, pad_W)


    def pecahan_gambar(self, citra_gambar):
        padded, info = self.padding(citra_gambar)
        H, W = padded.shape[:2]

        blocks = []
        blocks_shape = (H // 8, W // 8)

        for r in range(blocks_shape[0]):
            for c in range(blocks_shape[1]):
                blocks.append(padded[r*8 : r*8+8, c*8 : c*8+8])

        return blocks, padded.shape, info, blocks_shape


    def merge_blocks(self, blocks, shape, padded_info, blocks_shape):
        gambar = np.zeros(shape, dtype=np.uint8)
        
        in_block = 0
        for r in range(blocks_shape[0]):
            for c in range(blocks_shape[1]):
                gambar[r*8 : r*8+8, c*8 : c*8+8] = cv2.merge(blocks[in_block])
                in_block += 1

        gambar_bgr = cv2.cvtColor(gambar, cv2.COLOR_YCrCb2BGR)
        
        pad_H, pad_W = padded_info
        return gambar_bgr[:shape[0] - pad_H, :shape[1] - pad_W]


    def dct(self, blocks):
        dct_blocks = []

        for block in blocks:
            Y = block[0]
            Cr = block[1]
            Cb = block[2]
            dct_blocks.append([Y, cv2.dct(Cr.astype(np.float64)), cv2.dct(Cb.astype(np.float64))])
            
        return dct_blocks

    def quantize_blocks(self, dct_blocks, Q_TABLE):
        new_block = dct_blocks
        
        for block in new_block:
            block[1] = np.round(block[1] / Q_TABLE)
            block[2] = np.round(block[2] / Q_TABLE)

        return new_block

    def dequantize_blocks(self, quantized_blocks, Q_TABLE):
        new_block = quantized_blocks
        for block in new_block:
            block[1] =  block[1] * Q_TABLE
            block[2] =  block[2] * Q_TABLE

        return new_block

    def idct(self, blocks):
        idct_blocks = []
        for block in blocks:
            # block: [Y_deq, Cr_deq, Cb_deq] floats
            Y = block[0]
            Cr_idct = cv2.idct(block[1])
            Cb_idct = cv2.idct(block[2])

            # Clip to 0-255 and convert to uint8
            Cr_uint8 = np.clip(np.round(Cr_idct), 0, 255).astype(np.uint8)
            Cb_uint8 = np.clip(np.round(Cb_idct), 0, 255).astype(np.uint8)

            idct_blocks.append([Y, Cr_uint8, Cb_uint8])
        return idct_blocks

    def pecah_channel(self, blocks):
        blocks_split_channel = []
        for block in blocks:
            Y, Cr, Cb = cv2.split(block)
            blocks_split_channel.append([Y, Cr, Cb])
        
        return blocks_split_channel

    def set_lsb_safe(self, coeff, bit):
        coeff_int = int(round(coeff)) 
        if coeff_int >= 0:
            return ((abs(coeff_int) >> 1) << 1) | bit
        else:
            return -(((abs(coeff_int) >> 1) << 1) | bit)

        
    def get_lsb_safe(self, coeff):
        return abs(int(round(coeff))) & 1

