import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap,cm
import os
import warnings
from ftplib import FTP
import urllib.request

warnings.filterwarnings("ignore", category=UserWarning)



def Download(num_data):
    
    def grabFile(filename,num_data): # FTP download and save function
        # if the file already exists, skip it
        if os.path.isfile(local_dir+'/'+filename):
            return

        localfile = open(local_dir+'/'+filename, 'wb') # local download
        ftp.retrbinary('RETR ' + filename, localfile.write, 1024) # FTP saver

    ftp = FTP("ftp.bou.class.noaa.gov")
    ftp.login()
    ftp.cwd("./"+num_data1)
    files = ftp.nlst()

    local_dir = './GOES_files_'+num_data[:-4]
    if os.path.isdir(local_dir)==False:
        os.mkdir(local_dir)

    # loop through files and save them one-by-one
    for filename in files:
        grabFile(filename,num_data1)

    ftp.quit() # close FTP connection


num_data = ["6493387624/001","6493395794/001","6493405394/001","6493405334/001",
            "6493406414/001","6493411374/001","6493411384/001","6493418694/001",
            "6493418704/001","6493411504/001","6493413074/001","6498636844/001",
            "6498636854/001"]

for i in num_data:
    Download(i)
