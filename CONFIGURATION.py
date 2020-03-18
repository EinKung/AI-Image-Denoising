'''
Setting file
'''
import cv2,os,sys

# UI Files Directory
icon_path=os.path.join(sys.path[0],r'DATA\UIFiles\CLASSIFICATION.ico')
icon_exit_path=os.path.join(sys.path[0],r'DATA\UIFiles\exit.png')

# Data Directory
sample_dir=os.path.join(sys.path[0],r'DATA\SAMPLE')
train_dir=os.path.join(sys.path[0],r'DATA\TRAIN')
test_dir=os.path.join(sys.path[0],r'DATA\TEST')
cache_dir=os.path.join(sys.path[0],r'DATA\CACHE')
logFile_path=os.path.join(sys.path[0],r'DATA\launch.log')

# Training Setting
sampling_srgan_num_workers=4
sampling_idgan_num_workers=4
batch_size_srgan=64
batch_size_idgan=64
interval_srgan=10
interval_idgan=10
num_vggLayer_idgan=4
num_vggLayer_srgan=31

crop_size_esr=128
crop_size_id=256
up_scale=4
interpolation=cv2.INTER_CUBIC

lr_srgan=0.0002
lr_idgan=0.0002

alphaADV = 0.5
alphaPIX = 1.0
alphaFEA= 1.0
alphaSMO = 0.0001

alphaADV_ESR=0.001
alphaPIX_ESR=1.0
alphaFEA_ESR=0.006
alphaSMO_ESR=1.0

# Network Saved Path
network_srg_path=os.path.join(sys.path[0],r'DATA\NETWORK\generator_srgan.net')
network_idg_path=os.path.join(sys.path[0],r'DATA\NETWORK\generator_idgan.net')
network_srd_path=os.path.join(sys.path[0],r'DATA\NETWORK\discriminator_srgan.net')
network_idd_path=os.path.join(sys.path[0],r'DATA\NETWORK\discriminator_idgan.net')

# Degree of Pollution
dop_salt=[95,90,85,80]
dop_gauss=[15,25,35,45]
dop_poisson=[1.,15.,30.,50.]

# Completion Check List
checkList=[icon_path,
           icon_exit_path,
           network_srg_path,
           network_idg_path,
           os.path.join(sys.path[0],r'DATA\UIFiles\button_disable.png'),
           ]
