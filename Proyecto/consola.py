import os
from ftplib import FTP

def grabfile(filename):
    if os.path.isfile(local_dir+"/"+filename):
        return 
    localfile = open(local_dir+"/"+filename,"wb")
    ftp.retrbinary("RETR"+filename, localfile.write,1024)
    localfile.close()
    
ftp = FTP("ftp.bou.class.noaa.gov")
ftp.login()
ftp.cwd("./6478076024/001/")

files = ftp.nlst()
print(files)


local_dir = "./GOES_files"
if os.path.isdir(local_dir)==False:
    os.mkdir(local_dir)
    
for filename in files:
    grabfile(filename)
    
ftp.quit()

https://download.bou.class.noaa.gov/download/6478076024/001/OR_ABI-L2-LSTC-M6_G16_s20191930001325_e20191930004098_c20191930004503.nc