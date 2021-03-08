# -*- coding: utf-8 -*-

import argparse

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pylab
import statsmodels.api as sm

from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--dataframe_file", type=str, default="data.csv")
args = parser.parse_args()

def getModelCoefficients(x, y):
    n = len(x)
    xy_sum = np.sum(x * y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x2_sum = np.sum(np.power(x, 2))
    x_mean_pow = np.power(x_mean, 2)

    b1 = (xy_sum - (n * x_mean * y_mean)) / (x2_sum - (n * x_mean_pow))
    b0 = y_mean - (b1 * x_mean)
    
    return (b0, b1)

def getEstimation(x, b0, b1):
    return b0 + (b1 * x)

def getSSE(x, y, b0, b1):
    errors = []
    
    for i in range(0, len(x)):
        y_est = getEstimation(x[i], b0, b1)
        err = y[i] - y_est
        errors.append(err)
    
    return np.sum(np.power(errors, 2))

def getSST(x, y, b0, b1):
    y_mean = np.mean(y)
    errors = []
    
    for i in range(0, len(x)):
        y_est = getEstimation(x[i], b0, b1)
        err = np.power((y_est - y_mean), 2)
        errors.append(err)
    
    return np.sum(errors)

def getSSR(sse, sst):
    return (sst - sse)

def getDeterminationCoefficient(ssr, sst):
    return (ssr / sst)

def getQME(sse, n):
    return (sse / (n - 2))

def getSe(qme):
    return np.sqrt(qme)

def getSb0(x, se):
    n = len(x)
    
    x_pow2 = np.power(x, 2)
    
    x_mean = np.mean(x)
    x_mean_pow2 = np.power(x_mean, 2)
    
    return (se * np.sqrt((1 / n) + (x_mean_pow2 / (np.sum(x_pow2) - (n * x_mean_pow2)))))

def getSb1(x, se):
    n = len(x)
    
    x_pow2 = np.power(x, 2)
    
    x_mean = np.mean(x)
    x_mean_pow2 = np.power(x_mean, 2)
    
    return (se / (np.sqrt(np.sum(x_pow2) - (n * x_mean_pow2))))

def getConfidenceInterval(bi, sbi, alpha, df):
    t_val = stats.t.ppf(1 - (alpha / 2), df)
    #z_val = stats.norm.ppf(((alpha + 1) / 2))
    inter = t_val * sbi
    
    return (bi - inter, bi + inter)

def plotData(x, y, y_pred, title, x_label, y_label, saveFig=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    plt.plot(x, y_pred, c='r')
    
    if saveFig is not None:
        plt.savefig(os.path.join(args.output_dir, saveFig), dpi=200)
        
    plt.show()
    
def plotErrors(x, y, b0, b1, title, x_label, y_label):
    y_est = []
    errors = []
    
    for i in range(0, len(x)):
        y_i = getEstimation(x[i], b0, b1)
        err = y[i] - y_i
        
        y_est.append(y_i)
        errors.append(err)
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(y_est, errors)
    plt.savefig(os.path.join(args.output_dir, title + "_Estimativa x Erro da estimativa"), dpi=200)
    plt.show()
    
def qqPlot(x, y, b0, b1, title=None):
    errors = []
    
    for i in range(0, len(x)):
        y_est = getEstimation(x[i], b0, b1)
        err = y[i] - y_est
        errors.append(err)
        
    sm.qqplot(np.asarray(errors), line="q")
    plt.savefig(os.path.join(args.output_dir, title  + "_QQPlot"), dpi=200)
    pylab.show()
    
def rmse(targets, predictions):
    return np.sqrt(((np.asarray(predictions) - np.asarray(targets)) ** 2).mean())
    
def analyze(x_data, y_data, title=None, xlabel=None, ylabel=None):
    data = [x_data, y_data]
    data_len = x_data.size
    
    # Coeficiente de correlação
    corrcoef = np.corrcoef(data)
    
    # Parâmetros do modelo
    b0, b1 = getModelCoefficients(x_data, y_data)
    
    # Erros do modelo
    sse = getSSE(x_data, y_data, b0, b1)
    sst = getSST(x_data, y_data, b0, b1)
    ssr = getSSR(sse, sst)
    
    # Coeficiente de determinação (Qualidade do modelo)
    detcoef = getDeterminationCoefficient(ssr, sst)
    
    # Média dos erros ao quadrado
    qme = getQME(sse, data_len)
    
    # Desvio padrão dos erros
    se = getSe(qme)
    
    # Desvio padrão para os parâmetros b0 e b1
    sb0 = getSb0(x_data, se)
    sb1 = getSb1(x_data, se)
    
    # Intervalos de confiança para b0 e b1
    df = (data_len - 2)
    
    # 90% de confiança
    b0_inter_90 = getConfidenceInterval(b0, sb0, 0.10, df)
    b1_inter_90 = getConfidenceInterval(b1, sb1, 0.10, df)
    
    # 95% de confiança
    b0_inter_95 = getConfidenceInterval(b0, sb0, 0.05, df)
    b1_inter_95 = getConfidenceInterval(b1, sb1, 0.05, df)
    
    # 99% de confiança
    b0_inter_99 = getConfidenceInterval(b0, sb0, 0.01, df)
    b1_inter_99 = getConfidenceInterval(b1, sb1, 0.01, df)
        
    plotData(x_data, y_data, [getEstimation(x, b0, b1) for x in x_data],
             title, xlabel, ylabel,
             title)
    
    plotErrors(x_data, y_data, b0, b1,
             title, "Estimativa", "Erro da estimativa")
    
    qqPlot(x_data, y_data, b0, b1, title)
    
    return (corrcoef, 
            b0, b1, 
            sse, sst, ssr, 
            detcoef, 
            qme, se, sb0, sb1, 
            b0_inter_90, b1_inter_90,
            b0_inter_95, b1_inter_95,
            b0_inter_99, b1_inter_99)
    
df = pd.read_csv(os.path.join(args.output_dir, args.dataframe_file), sep=';')
df = df.loc[df["DMOS"] != 0]

dmos_data = np.asarray(df["DMOS"])
psnr_data = np.asarray(df["PSNR"])
ssim_data = np.asarray(df["SSIM"])
msssim_data = np.asarray(df["MS-SSIM"])

x_psnr_data = np.sort(psnr_data)
x_ssim_data = np.sort(ssim_data)[::-1]
x_msssim_data = np.sort(msssim_data)[::-1]
y_dmos_data = np.sort(dmos_data)

(psnr_corrcoef, 
psnr_b0, psnr_b1, 
psnr_sse, psnr_sst, psnr_ssr, 
psnr_detcoef, 
psnr_qme, psnr_se, psnr_sb0, psnr_sb1, 
psnr_b0_inter_90, psnr_b1_inter_90,
psnr_b0_inter_95, psnr_b1_inter_95,
psnr_b0_inter_99, psnr_b1_inter_99) = analyze(x_psnr_data, y_dmos_data, "PSNR x DMOS", "PSNR", "DMOS")

(ssim_corrcoef, 
ssim_b0, ssim_b1, 
ssim_sse, ssim_sst, ssim_ssr, 
ssim_detcoef, 
ssim_qme, ssim_se, ssim_sb0, ssim_sb1, 
ssim_b0_inter_90, ssim_b1_inter_90,
ssim_b0_inter_95, ssim_b1_inter_95,
ssim_b0_inter_99, ssim_b1_inter_99) = analyze(x_ssim_data, y_dmos_data, "SSIM x DMOS", "SSIM", "DMOS")

(msssim_corrcoef, 
msssim_b0, msssim_b1, 
msssim_sse, msssim_sst, msssim_ssr, 
msssim_detcoef, 
msssim_qme, msssim_se, msssim_sb0, msssim_sb1, 
msssim_b0_inter_90, msssim_b1_inter_90,
msssim_b0_inter_95, msssim_b1_inter_95,
msssim_b0_inter_99, msssim_b1_inter_99) = analyze(x_msssim_data, y_dmos_data, "MS-SSIM x DMOS", "MS-SSIM", "DMOS")

#df_filtred = df.loc[df["Original filename"] == "bikes.bmp"]
df_filtred = df

psnr_data = []
ssim_data = []
msssim_data = []

y_true = []
y_psnr_pred = []
y_ssim_pred = []
y_msssim_pred = []

for (index, row) in df_filtred.iterrows():
    psnr = row["PSNR"]
    ssim = row["SSIM"]
    msssim = row["MS-SSIM"]
    
    dmos = row["DMOS"]
    dmos_psnr_pred = getEstimation(psnr, psnr_b0, psnr_b1)
    dmos_ssim_pred = getEstimation(ssim, ssim_b0, ssim_b1)
    dmos_msssim_pred = getEstimation(msssim, msssim_b0, msssim_b1)
    
    y_true.append(dmos)
    y_psnr_pred.append(dmos_psnr_pred)
    y_ssim_pred.append(dmos_ssim_pred)
    y_msssim_pred.append(dmos_msssim_pred)
    
    psnr_data.append([row["Count"], row["Index"], row["Original filename"], row["Current filename"], 
                      psnr, dmos, dmos_psnr_pred, (dmos_psnr_pred - dmos)])
    
    ssim_data.append([row["Count"], row["Index"], row["Original filename"], row["Current filename"], 
                      ssim, dmos, dmos_ssim_pred, (dmos_ssim_pred - dmos)])
    
    msssim_data.append([row["Count"], row["Index"], row["Original filename"], row["Current filename"], 
                      msssim, dmos, dmos_msssim_pred, (dmos_msssim_pred - dmos)])
    
    print(row["Current filename"] 
            + " -- DMOS: " + str(dmos) 
            + " -- DMOS(PSNR): " + str(dmos_psnr_pred)
            + " -- DMOS(SSIM): " + str(dmos_ssim_pred)
            + " -- DMOS(MSSSIM): " + str(dmos_msssim_pred))


df_psnr_result = pd.DataFrame(psnr_data, columns=["Count", "Index", "Original filename", "Current filename", "PSNR", "DMOS", "DMOS(pred)", "Error"])
df_psnr_result.to_csv(os.path.join(args.output_dir, "result_psnr.csv"), index=None, sep=';')

df_ssim_result = pd.DataFrame(ssim_data, columns=["Count", "Index", "Original filename", "Current filename", "SSIM", "DMOS", "DMOS(pred)", "Error"])
df_ssim_result.to_csv(os.path.join(args.output_dir, "result_ssim.csv"), index=None, sep=';')

df_msssim_result = pd.DataFrame(msssim_data, columns=["Count", "Index", "Original filename", "Current filename", "MS-SSIM", "DMOS", "DMOS(pred)", "Error"])
df_msssim_result.to_csv(os.path.join(args.output_dir, "result_msssim.csv"), index=None, sep=';')

print()
print("RMSE(PSNR): " + str(rmse(y_true, y_psnr_pred)))
print("RMSE(SSIM): " + str(rmse(y_true, y_ssim_pred)))
print("RMSE(MSSSIM): " + str(rmse(y_true, y_msssim_pred)))