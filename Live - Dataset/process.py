# -*- coding: utf-8 -*-

import argparse

import os
import shutil

import scipy.io
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="jpeg")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--refnames_file", type=str, default="refnames_all.mat")
parser.add_argument("--dmos_file", type=str, default="dmos.mat")
parser.add_argument("--range_start", type=int, default=227)
parser.add_argument("--range_end", type=int, default=460)
parser.add_argument("--dataframe_file", type=str, default="data.csv")
args = parser.parse_args()

def read_info_file(path):
    info_list = []
    
    with open(path, "r") as file:
        for line in file:
            line_split = line.split(' ')
            
            orig_filename = line_split[0]
            curr_filename = line_split[1]
            bitrate = line_split[2]
            
            info_list.append([orig_filename, curr_filename, bitrate])
            
    return info_list    

# Faz a leitura dos dados originais do dataset (Arquivos do MatLab)
print("Fazendo a leitura dos dados originais do dataset")
refnames_all_file = scipy.io.loadmat(args.refnames_file)
dmos_file = scipy.io.loadmat(args.dmos_file)
info_file = read_info_file(os.path.join(args.dataset_dir, "info.txt"))

# Processa os dados originais para criar um dataframe com os dados
print("Processando os dados originais para criar um dataframe")
data = []

for i in range(args.range_start, args.range_end):
    count = i
    index = (i - args.range_start) + 1
    ref_filename = refnames_all_file["refnames_all"][0, i][0]
    curr_filename = "img" + str(index) + ".bmp"
    bitrate = info_file[index - 1][2]
    ref_value = dmos_file["orgs"][0, i]
    dmos_value = dmos_file["dmos"][0, i]
    psnr_value = 0
    ssim_value = 0
    msssim_value = 0
    
    data.append([count, index, ref_filename, curr_filename, bitrate, ref_value, dmos_value, psnr_value, ssim_value, msssim_value])

df = pd.DataFrame(data, columns=["Count", "Index", "Original filename", "Current filename", "Bitrate", "Reference", "DMOS", "PSNR", "SSIM", "MS-SSIM"])
df = df.sort_values(["Original filename", "Reference"])

# Processa os arquivos de imagem para organizar melhor
if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)

os.makedirs(args.output_dir)

print("Processando os arquivos de imagem para melhor organização")
original_filenames = df["Original filename"].unique().tolist()
df_filtred_list = []

for filename in original_filenames:
    df_filtred = df.loc[df["Original filename"] == filename]
    df_filtred_list.append(df_filtred)
    
    out_dirname = os.path.splitext(filename)[0]
    out_dirpath = os.path.join(args.output_dir, out_dirname)
    
    os.mkdir(out_dirpath)
    
    for (index, row) in df_filtred.iterrows():
        srcFile = os.path.join(args.dataset_dir, row["Current filename"])
        dstFile = os.path.join(out_dirpath, row["Current filename"])
        
        shutil.copy2(srcFile, dstFile)

# Obtém as métricas de qualidade perceptual para os conjuntos de imagens  
print("Obtendo as métricas de qualidade perceptual para os conjuntos de imagens")

for df_filtred in df_filtred_list:
    ref_img_path = os.path.join(args.dataset_dir, df_filtred.iloc[-1]["Current filename"])
    print(ref_img_path)
    
    ref_img = tf.keras.preprocessing.image.img_to_array(
                        tf.keras.preprocessing.image.load_img(ref_img_path))
    ref_img = tf.image.convert_image_dtype(ref_img, tf.uint8)
        
    for i in range(0, 7):
        dist_img_path = os.path.join(args.dataset_dir, df_filtred.iloc[i]["Current filename"])
        print(dist_img_path)
        
        dist_img = tf.keras.preprocessing.image.img_to_array(
                            tf.keras.preprocessing.image.load_img(dist_img_path))
        dist_img = tf.image.convert_image_dtype(dist_img, tf.uint8)
        
        psnr = tf.image.psnr(ref_img, dist_img, max_val=255).numpy()
        ssim = tf.image.ssim(ref_img, dist_img, max_val=255).numpy()
        msssim = tf.image.ssim_multiscale(ref_img, dist_img, max_val=255).numpy()
        
        index = (df_filtred.iloc[i]["Index"] - 1)
        
        df.loc[index, "PSNR"] = psnr
        df.loc[index, "SSIM"] = ssim
        df.loc[index, "MS-SSIM"] = msssim
    
    print()
        
# Salva o dataframe em um arquivo CSV       
df.to_csv(os.path.join(args.output_dir, args.dataframe_file), index=None, sep=';')
print("Dataframe salvo em arquivo CSV")