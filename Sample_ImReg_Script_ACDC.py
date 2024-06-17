import numpy as np
import nibabel as nib
# from aux_functions_time import *
from EDITED_aux_functions_time import *
from WarpPINN import WarpPINN
import time
import os
import matplotlib.pyplot as plt
import glob
import pyvista as pv
from skimage import measure
import meshio
import vtk
import time
from tqdm import tqdm
import cv2
import tetgen
from skimage.measure import label, regionprops
import pandas as pd
from scipy.ndimage import map_coordinates
import pyacvd


#################################################################################
## Hyperparameter Tuning ##

patient_number = 1

# mu_NeoHook = 5e-6
# mu_NeoHook = 1e-5
# mu_NeoHook = 5e-4
# mu_NeoHook = 5e-3

mu_NeoHook = 5e-2

lmbmu = 10**8

#################################################################################




mu_NeoHook_str = str(mu_NeoHook)



patients = os.listdir('ACDC_raw/database/training')
if '.ipynb_checkpoints' in patients:
    patients.remove('.ipynb_checkpoints')
if 'ACDC_training_patient_info.csv' in patients:
    patients.remove('ACDC_training_patient_info.csv') 
if 'MANDATORY_CITATION.md' in patients:
    patients.remove('MANDATORY_CITATION.md')

patients.sort()

patient_info = pd.read_csv('ACDC_raw/database/training/ACDC_training_patient_info.csv')
patient_info = patient_info.set_index('patient', drop=True)

metrics_file = 'ACDC_raw/metrics.csv'
# patient_info.head(5)



if os.path.exists( metrics_file ):
    metrics = pd.read_csv( metrics_file )
    metrics = metrics.set_index('patient', drop = True)
    metrics.index.name = 'patient'
else:
    metrics = pd.DataFrame(index = patients , columns = ['MSE', 'SSIM', 'MCD', 'HD', 'Dice', '||J| - 1|', 'Group'])
    metrics.index.name = 'patient'

# metrics.head()
    

# idx_pt = 0  # patient index
idx_pt = patient_number - 1
patient = patients[idx_pt]
es_frame_no = patient_info['ES_frame'][patient]
data_path = os.path.join( 'ACDC_raw/database/training', patient)

imt_path = glob.glob(f'{data_path}/*4d.nii.gz')[0]
imt_data = nib.load(imt_path)
imt = imt_data.get_fdata()
header = imt_data.header

patient_no = patient[7:]
print(f'This is patient: {patient_no}')
print(f'The ES frame is: {es_frame_no}')

# Pixel spacing of image
pix_dim = header['pixdim']
pixsp_x = pix_dim[1]
pixsp_y = pix_dim[2]
pixsp_z = pix_dim[3]

imt = np.transpose(imt, [3, 2, 1, 0])

frames = imt.shape[0]
slices = imt.shape[1]



#Importing Ground-Truth Segmentation/End-Diastolic 

segm_imt_path = glob.glob(f'{data_path}/*frame01_gt.nii*')[0]
segm_imt_data = nib.load(segm_imt_path)
segm_imt = segm_imt_data.get_fdata()
segm_header = segm_imt_data.header
dim = segm_header['dim']
dims = segm_imt.shape
mask_pixdim = segm_header['pixdim']

# Create a binary mask of GT

low = 1.5 #
high = 3
mask = np.zeros(dims)
mask[(segm_imt > low) & (segm_imt < high)] = 1

#Importing End-Systolic Segmentation 

es_frame_no = int(patient_info['ES_frame'][patient])
print(f'ES Frame No: {es_frame_no}')
es_imt_path = glob.glob(f'{data_path}/*frame*{es_frame_no}*_gt.nii*')[0]
print('ES Frame Path: '+ es_imt_path)
es_imt_data = nib.load(es_imt_path)
es_imt = es_imt_data.get_fdata()
es_imt = np.transpose(es_imt, [2, 1, 0])


# Create a binary mask of ES frame

low = 1.5 
high = 3
es_mask = np.zeros(es_imt.shape)
es_mask[(es_imt > low) & (es_imt < high)] = 1

#Crop mask 

s=0
img = mask[:, :, s]
labeled_img = label(img)
regions = regionprops(labeled_img)

if len(regions) == 0:
    s=1 
    img = mask[:, :, s]
    labeled_img = label(img)
    regions = regionprops(labeled_img)

myocardium_region = max(regions, key=lambda x: x.area)

# bounding box of myo
min_x, min_y, max_x, max_y = myocardium_region.bbox

# expand bounding box 5 pix in each direction if possible
buffer = 5 
min_x = max(0, min_x - buffer)
min_y = max(0, min_y - buffer)
max_x = min(img.shape[0], max_x + buffer)
max_y = min(img.shape[1], max_y + buffer)


for i in range(s+1,dims[2]): 
    img = mask[:, :, i]
    labeled_img = label(img)
    regions = regionprops(labeled_img)
    
    if len(regions) == 0:
        continue
    
    myocardium_region = max(regions, key=lambda x: x.area)

    # bounding box of myo
    minr, minc, maxr, maxc = myocardium_region.bbox


    # expand bounding box 20 pix in each direction if possible
    minr = max(0, minr - buffer)
    minc = max(0, minc - buffer)
    maxr = min(img.shape[0], maxr + buffer)
    maxc = min(img.shape[1], maxc + buffer)

    if minr < min_x:
        min_x = minr
    if maxr > max_x:
        max_x = maxr
    if minc < min_y:
        min_y = minc
    if maxc > max_y:
        max_y = maxc

def round_to_even(number):
    rounded_number = round(number)
    if rounded_number % 2 != 0:  # Check if the number is odd
        rounded_number += 1     # Increment by 1 to make it even
    return rounded_number

crop_x_in = round_to_even(min_x)
crop_x_end = round_to_even(max_x)
crop_y_in = round_to_even(min_y)
crop_y_end = round_to_even(max_y)

crop_str = str(crop_x_in)+'_'+str(crop_x_end)+'_'+str(crop_y_in)+'_'+str(crop_y_end)


imt = imt[:, :, crop_y_in:crop_y_end, crop_x_in:crop_x_end].astype(np.float32)

# Reference image is the first time in the stack

imr = imt[0, :, :, :]


pad_mask = np.pad(mask, ((0, 0), (0, 0), (1, 1)), mode='constant')

Nx = dim[1]
Ny = dim[2]
Nz = dim[3]

xp = np.arange(Nx)
yp = np.arange(Ny)
zp = np.arange(Nz)

Y, Z, X = np.meshgrid(yp, zp, xp)

X = X[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
Y = Y[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
Z = Z[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]

X = X.ravel()[:,None]
Y = Y.ravel()[:,None]
Z = Z.ravel()[:,None]

# Pixel coordinates in a list
im_pix = np.hstack((X, Y, Z))
im_mesh = pixel_to_mesh(im_pix, pixsp_x, pixsp_y, pixsp_z)








#Surface Mesh Generation

# Define scaling factors for each axis
scale_x = mask_pixdim[1] # Example scaling factor for the x-axis
scale_y = mask_pixdim[2] # Example scaling factor for the y-axis
scale_z = mask_pixdim[3] # Example scaling factor for the z-axis

vertices, triangles, norm, val = measure.marching_cubes(volume=pad_mask,
                                                    level=1e-12,
                                                    spacing=(scale_x, scale_y, scale_z),
                                                    gradient_direction='descent',
                                                    step_size=1,
                                                    allow_degenerate=False,
                                                    method='lewiner')


# Create a pyvista mesh object using the vertices and triangles
mesh = pv.PolyData(vertices)
faces = np.hstack((np.full((len(triangles), 1), 3), triangles)).ravel()
# Create the cell type array (VTK_TRIANGLE = 5)
cell_types = np.full(len(triangles), 5)
# create the unstructured grid directly from the numpy arrays
surface = pv.UnstructuredGrid(faces, cell_types, vertices)

#Generating Volume Mesh
try:
    faces = surface.cell_connectivity.reshape(-1,3)
    tet = tetgen.TetGen(surface.points, faces)
    tet.make_manifold()
    tet.tetrahedralize(mindihedral=25, minratio = 1.1)
    volume = tet.grid
except RuntimeError:
    print(f'Cannot make volume mesh for patient {patient_no}! Moving to next patient.')

#Smoothing Surface Mesh

mesh = surface.extract_surface()
mesh.triangulate().clean(tolerance=1e-8)
mesh.subdivide_adaptive(max_n_passes=100, max_edge_len=1.0)
# Volume preserving smoothing
mesh.smooth_taubin(n_iter=100, pass_band=0.75, edge_angle=30, feature_angle=45,
                    boundary_smoothing=True, feature_smoothing=True,
                    non_manifold_smoothing=True, normalize_coordinates=False,
                    inplace=True)
# Small Laplacian smoothing to remove outliers
mesh.smooth(n_iter=10, edge_angle=150, feature_angle=180, feature_smoothing=True,
            relaxation_factor=0.1, convergence=0.1, inplace=True)

surface = mesh.clean(tolerance=1e-8)

# ACVD: Clustering -> Remeshing
clus = pyacvd.Clustering(surface)
clus.subdivide(3)
clus.cluster(20000)

surface = clus.create_mesh()

surface = surface.cast_to_unstructured_grid()

#Save surface and volume meshes

volume.save(f'ACDC_raw/database/training/patient{patient_no}/patient{patient_no}_vol.vtu')
surface.save(f'ACDC_raw/database/training/patient{patient_no}/patient{patient_no}_surf_vol.vtu')


# Mesh coords of image voxels and segmentation nodes
im_mesh, segm_mesh = im_and_segm_mesh(imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, data_path)

# Mesh coords for background nodes
bg_mesh_file = os.path.join(data_path, 'background_points_'+crop_str+'.npy')
bg_mesh = background_mesh(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, bg_mesh_file, slices)

# Boolean mask: 1 if the voxel is in the LV, 0 otherwise
bool_mask_file = os.path.join(data_path, 'boolean_mask_'+crop_str+'.npy')
bool_mask = boolean_mask(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, bool_mask_file )

### Neural Network ###

# Layers for u1, u2 and u3
neu = 2**6
layers_u = [4, neu, neu, neu, neu, neu, 3]

reg_mask = 1
lmbmu = lmbmu
# mu_NeoHook = 1e-5

# Number of iterations desired for the registration from t_1 to t_i
It = 300000             
batch_size = 1000
# Number of points in LV used to calculate the Neo Hookean by minibatch
N_nodes = len(segm_mesh)
# Number of epochs 
nEpoch = int( np.ceil(It * batch_size / N_nodes) )

pix_crop = [pixsp_x, pixsp_y, pixsp_z, crop_x_in, crop_y_in]



#################################################
# Saving:
    # Volume mesh 
    # Suface mesh
    # Background mesh
    # Boolean mask 
    # Image mesh
    # Segmentation mesh



ACDC_save_path = os.path.join(f'HyperparameterTuning_ACDC_Baseline_Patient{patient_number}_mu{mu_NeoHook_str}_lmbmu{lmbmu}_Results')
ACDC_save_path_generatedMeshes = os.path.join(ACDC_save_path, 'Saved_generated_meshes')

# ACDC_save_path = os.path.join('ACDC_Baseline_Patient1_ImprovedMeshing_Results')
# ACDC_save_path_generatedMeshes = os.path.join('ACDC_Baseline_Patient1_ImprovedMeshing_Results', 'Saved_generated_meshes')

if not os.path.exists(ACDC_save_path):
    os.makedirs(ACDC_save_path)
if not os.path.exists(ACDC_save_path_generatedMeshes):
    os.makedirs(ACDC_save_path_generatedMeshes)

volume.save( os.path.join(ACDC_save_path_generatedMeshes, 'vol_mesh.vtu') )
surface.save( os.path.join(ACDC_save_path_generatedMeshes, 'surf_vol_mesh.vtu') )

# volume.save('ACDC_Baseline_Patient1_ImprovedMeshing_Results\\Saved_generated_meshes\\vol_mesh.vtu')
# surface.save('ACDC_Baseline_Patient1_ImprovedMeshing_Results\\Saved_generated_meshes\\surf_vol_mesh.vtu')


np.save( os.path.join( ACDC_save_path, 'im_mesh.npy' ), im_mesh )
np.save( os.path.join( ACDC_save_path, 'segm_mesh.npy' ), segm_mesh )
np.save( os.path.join( ACDC_save_path, 'bg_mesh.npy' ), bg_mesh )
np.save( os.path.join( ACDC_save_path, 'bool_mask.npy' ), bool_mask )

##################################################



model = WarpPINN(imr, imt, layers_u, bool_mask, im_mesh, segm_mesh, bg_mesh, lmbmu, pix_crop, reg_mask
                 , simpleitk_transform=False)






# Directory savings:

LossCurve_save_path = os.path.join(ACDC_save_path, 'LossCurve.csv')
WeightsAndBiases_save_path = os.path.join(ACDC_save_path, 'WeightsAndBiases.json')





t0 = time.time()
tol_pretrain = 1e-6
model.pretrain(tol_pretrain)
t1 = time.time()



es_imt_path = glob.glob(f'{data_path}/*frame1*_gt.nii*')[0]
es_imt_data = nib.load(es_imt_path)
es_imt = es_imt_data.get_fdata()
es_imt = np.transpose(es_imt, [2, 1, 0])

es_frame_no = int(es_imt_path[54:56])

low = 1.5 
high = 3
es_mask = np.zeros(es_imt.shape)
es_mask[(es_imt > low) & (es_imt < high)] = 1
es_mask = es_mask[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end].astype(np.float32)


try:
    # Choose either L1 norm or L2 norm to measure the difference between reference and warped template images

    # model.train_Adam_L1_NeoHook(nEpoch, mu_NeoHook, size=batch_size)
    model.train_Adam_L1_NeoHook(nEpoch, mu_NeoHook, LossCurve_save_path, size=batch_size
                                , EarlyStopping=False) # Trying to train with just 1/few epochs to see if code runs...
    #model.train_Adam_MSE_NeoHook(nEpoch, mu_NeoHook, size=batch_size) 

except:
    
    print('J = 0 at some point. Training has stopped. Predictions are made from last iteration.')

finally:

    ### For HPC saving ###
    model.save_WeightsAndBiases(WeightsAndBiases_save_path)

    ### Save model ###

    str_neu = str(neu)
    str_mask = str(reg_mask)
    str_batch = str(batch_size)

    str_mu_NeoHook = str(mu_NeoHook)
    str_lmbmu = str(lmbmu)

    model_name = 'model_L1_'+str_mu_NeoHook+'_NH_lmbmu_'+str_lmbmu+'_mask_'+str_mask+'_neu_'+str_neu+'_batch_'+str_batch
    # model_name = 'model_L2_'+str_mu_NeoHook+'_NH_lmbmu_'+str_lmbmu+'_mask_'+str_mask+'_neu_'+str_neu+'_batch_'+str_batch

    model_dir = os.path.join( ACDC_save_path, 'Validation_Results', patient, model_name) 
    # model_dir = os.path.join( 'results', 'L1', patient, model_name) 
    #model_dir = os.path.join( 'results', 'L2', volunteer, model_name) 

    model_save = os.path.join(model_dir, model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.saver.save(model.sess, model_save)

    t2 = time.time()    

    ### Loading data for post-processing ###

    # Mesh coords of nodes in the ventricle to be deformed to simulate heart beating
    surf_mesh = meshio.read( os.path.join( data_path, patient +'_surf_vol.vtu' ))


    # For strains

    # Mesh coords
    origin = np.mean(surf_mesh.points, axis = 0)
    cart_coords = surf_mesh.points - origin
    angle = np.arctan2(cart_coords[:,1], cart_coords[:,0])

    rho = np.hstack((np.cos(angle)[:,None], np.sin(angle)[:,None]))
    theta = np.hstack((-np.sin(angle)[:,None], np.cos(angle)[:,None]))

    rho_1 = np.reshape(rho, [-1,1,2])
    rho_2 = np.reshape(rho, [-1,2,1])

    theta_1 = np.reshape(theta, [-1,1,2])
    theta_2 = np.reshape(theta, [-1,2,1])

    # Surface at first frame
    surf_path = os.path.join(model_dir, 'surfaces')
    if not os.path.exists(surf_path):
        os.makedirs( surf_path )

    num_nodes = len(surf_mesh.points)

    file_name = os.path.join(surf_path, 'p001_surf.vtk')
    meshio.write(file_name, meshio.Mesh(points=surf_mesh.points, cells = {'triangle': surf_mesh.cells_dict['triangle']}, \
                    point_data = {'radStrain':np.zeros([num_nodes]), 'circStrain':np.zeros([num_nodes]), 'longStrain': np.zeros([num_nodes]), 'Jacobian': np.ones([num_nodes])}))

    # Landmarks at first frame - There are no landmarks in ACDC?
    lmks_path = os.path.join(model_dir, 'lmks')
    if not os.path.exists( lmks_path ):
        os.makedirs( lmks_path )


    # To generate some basic plots for Reference image vs Predicted image
    mse_pred_list = []
    ssim_pred_list = []
    dice_pred_list = []

    # To generate some basic plots for Reference image vs Template image
    mse_temp_list = []
    ssim_temp_list = []
    dice_temp_list = []
    
     # To generate some basic plots for Template image vs Warped image
    mse_wartemp_list = []
    ssim_wartemp_list = []
    dice_wartemp_list = []
    
    #TESTING DICE SCORE
    def dice_score(true, pred):
        intersection = np.logical_and(true, pred).sum()
        total_pixels = np.size(true) + np.size(pred)
        dice = (2.0 * intersection) / total_pixels if total_pixels != 0 else 0.0
        return dice

    # To generate some basic plots for strains
    # Radial
    mean_rr = np.zeros([frames])
    median_rr = np.zeros([frames])
    std_rr = np.zeros([frames])
    # Circumferential
    mean_cc = np.zeros([frames])
    median_cc = np.zeros([frames])
    std_cc = np.zeros([frames])
    # Longitudinal
    mean_ll = np.zeros([frames])
    median_ll = np.zeros([frames])
    std_ll = np.zeros([frames])
    
    imt_resc2 = np.zeros(np.shape(imt))
    imt_pred2 = np.zeros(np.shape(imt))
    
    d1 = []
    d2 = []
    d3 = []

    # Warping at different times

    ### =============SAVING IM_STARS and DEFORMATION FIELD COMPONENTS======================
    im_star_array = []
    u1_star_array = []
    u2_star_array = []
    u3_star_array = []
    ### ===================================================================================




    for i in range(frames):

        # Surface warp
        surf_mesh_warp, u1, u2, u3, u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z, J = model.surface_deformation(surface.points, i)

        grad_um = np.hstack((u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z))
        grad_um = np.reshape(grad_um, [-1, 3, 3])
        grad_um_t = np.transpose(grad_um, [0, 2, 1])

        Em = 1/2 * (grad_um + grad_um_t + np.matmul(grad_um_t, grad_um))

        Em_new = Em[:,0:2,0:2]
        # Radial strain
        Em_rr = np.matmul(rho_1, np.matmul(Em_new, rho_2))
        # Circumferential strain
        Em_cc = np.matmul(theta_1, np.matmul(Em_new, theta_2))
        # Longitudinal strain
        Em_ll = Em[:,2,2]

        mean_rr[i] = np.mean(Em_rr)
        median_rr[i] = np.median(Em_rr)
        std_rr[i] = np.std(Em_rr)

        mean_cc[i] = np.mean(Em_cc)
        median_cc[i] = np.median(Em_cc)
        std_cc[i] = np.std(Em_cc)

        mean_ll[i] = np.mean(Em_ll)
        median_ll[i] = np.median(Em_ll)
        std_ll[i] = np.std(Em_ll)

        # Save warped surface with strains and jacobian
        file_name = os.path.join(surf_path, 'p{0}_surf.vtk'.format(i+1))
        meshio.write(file_name, meshio.Mesh(points=surf_mesh_warp, cells = {'triangle': surf_mesh.cells_dict['triangle']}, \
                    point_data = {'radStrain':Em_rr, 'circStrain':Em_cc, 'longStrain':Em_ll, 'Jacobian': J}))
        

        # Prediction on voxel coordinates

        imr_pred, u1_pred, u2_pred, u3_pred, u1x_pred, u1y_pred, u1z_pred, u2x_pred, u2y_pred, u2z_pred, u3x_pred, u3y_pred, u3z_pred, J_pred = model.predict(i)
        



        im_star_array.append(imr_pred)
        u1_star_array.append(u1_pred)
        u2_star_array.append(u2_pred)
        u3_star_array.append(u3_pred)




        d1.append(u1_pred)
        d2.append(u2_pred)
        d3.append(u3_pred)

        imt_i = imt[i,:,:,:]
        ub_im = np.max([np.max(imr), np.max(imt)])
        lb_im = np.min([np.min(imr), np.min(imt)])
        imr_resc = (imr - lb_im) / (ub_im - lb_im)
        imt_resc = (imt_i - lb_im) / (ub_im - lb_im)

        # Comparing predicted image and reference image
        print('Reference image vs predicted image frame {0}'.format(i))
        compare_images(imr_resc[1:-1], imr_pred[1:-1])
        print('Reference image vs Template image frame {0}'.format(i))
        compare_images(imr_resc[1:-1], imt_resc[1:-1])

        mse_pred_list.append( np.sqrt((np.square(imr_resc[1:-1]-imr_pred[1:-1])).mean() / (np.square(imr_resc[1:-1])).mean()) )
        mse_temp_list.append( np.sqrt((np.square(imr_resc[1:-1]-imt_resc[1:-1])).mean() / (np.square(imr_resc[1:-1])).mean()) )
        mse_wartemp_list.append( np.sqrt((np.square(imt_resc[1:-1]-imr_pred[1:-1])).mean() / (np.square(imr_resc[1:-1])).mean()) )
        
        ssim_pred_list.append( ssim(imr_resc[1:-1], imr_pred[1:-1], data_range=imr_resc[1:-1].max() - imr_resc[1:-1].min())) #added data range to ssim method
        ssim_temp_list.append( ssim(imr_resc[1:-1], imt_resc[1:-1], data_range=imr_resc[1:-1].max() - imr_resc[1:-1].min())) #added data range to ssim method
        ssim_wartemp_list.append( ssim(imt_resc[1:-1], imr_pred[1:-1], data_range=imr_resc[1:-1].max() - imr_resc[1:-1].min())) 
        
        dice_pred_list.append( dice_score(imr_resc[1:-1], imr_pred[1:-1]))
        dice_temp_list.append( dice_score(imr_resc[1:-1], imt_resc[1:-1]))
        dice_wartemp_list.append( dice_score(imt_resc[1:-1], imr_pred[1:-1]))
        
        imt_resc2[i,:,:,:] = imt_resc
        imt_pred2[i,:,:,:] = imr_pred






    np.save( os.path.join(model_dir, 'im_star_array.npy') , im_star_array )
    np.save( os.path.join(model_dir, 'u1_star_array.npy') , u1_star_array )
    np.save( os.path.join(model_dir, 'u2_star_array.npy') , u2_star_array )
    np.save( os.path.join(model_dir, 'u3_star_array.npy') , u3_star_array )


    np.save( os.path.join(model_dir, 'mse_pred_list.npy'), np.array(mse_pred_list))
    np.save( os.path.join(model_dir, 'mse_temp_list.npy'), np.array(mse_temp_list))
    np.save( os.path.join(model_dir, 'mse_wartemp_list.npy'), np.array(mse_wartemp_list))

    np.save( os.path.join(model_dir, 'ssim_pred_list.npy'), np.array(ssim_pred_list))
    np.save( os.path.join(model_dir, 'ssim_temp_list.npy'), np.array(ssim_temp_list))
    np.save( os.path.join(model_dir, 'ssim_wartemp_list.npy'), np.array(ssim_wartemp_list))

    np.save( os.path.join(model_dir, 'dice_pred_list.npy'), np.array(dice_pred_list))
    np.save( os.path.join(model_dir, 'dice_temp_list.npy'), np.array(dice_temp_list))
    np.save( os.path.join(model_dir, 'dice_wartemp_list.npy'), np.array(dice_wartemp_list))





    np.save( os.path.join(model_dir, 'loss_registration.npy'), np.array(model.lossit_MSE))
    np.save( os.path.join(model_dir, 'loss_nh.npy'), np.array(model.lossit_NeoHook))
    np.save( os.path.join(model_dir, 'loss.npy'), np.array(model.lossit_value))