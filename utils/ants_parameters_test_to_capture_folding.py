import os
# Script to test ANTS registration to capture brain folding
# Input intensity images : t2_t<week>.00_128.nii.gz
# Input segmented images : t2-t<week>.00_128-bounti_label_{label}.nii.gz

# Current best result obtained with :
# segmentation only (all labels)
# transform = SyN
# gradientStep = 1

# 21 weeks is considered as a fixed starting point
starting_week = 21
last_week = 36
# List of weeks to process
weeks = range(36,last_week+1,1)
path_labels = '/home/florian/Documents/Dataset/dHCP/parcellations/binary_mask/'
path_intensity = '/home/florian/Documents/Dataset/dHCP/structural/'

path_intensity_starting = os.path.join(path_intensity, f't2-t{starting_week}.00.nii.gz')
path_labels_starting = os.path.join(path_labels, f't{starting_week}.00.nii.gz')
use_seg = True
use_intensity = False

labels = 'all'

output_path = './outputs/'
os.makedirs(output_path, exist_ok=True)

if labels == 'cortex':
    label_range = [3, 4]
elif labels == 'cortex_ventricles':
    label_range = [3, 4, 7, 8]
elif labels == 'all':
    label_range = range(0, 20)

# ANTs registration parameters
loss = 'CC'  # loss sur intensité, CC ou MI
gradientSteps = ['1']  # possibility to test a range of values

transform = 'SyN'
# transform = 'BSplineSyN'
# transform = 'TimeVaryingVelocityField'
# transform = 'TimeVaryingBSplineVelocityField'

path_intensity_starting = os.path.join(path_intensity, f't2-t{starting_week}.00.nii.gz')

for gradientStep in gradientSteps:

    # Loop over weeks
    for week in weeks:

        # Construct the command
        prefix = f'ants_{transform}_gs{gradientStep}'
        if use_seg is True:
            prefix += '_seg_' + labels
        if use_intensity is False:
            prefix += f'_{loss}_intensity'
        cmd = f'antsRegistration -d 3 -o {prefix} '

        if transform == 'SyN':
            # SyN[gradientStep,<updateFieldVarianceInVoxelSpace=3>,<totalFieldVarianceInVoxelSpace=0>]
            # efault values from antsMultivariateTemplateConstruction2.sh : [ 0.1,3,0 ]
            cmd += f'--transform {transform}[' + gradientStep + ',2,0] '
        elif transform == 'BSplineSyN':
            # BSplineSyN[gradientStep,updateFieldMeshSizeAtBaseLevel,<totalFieldMeshSizeAtBaseLevel=0>,<splineOrder=3>]
            # efault values from antsMultivariateTemplateConstruction2.sh : [ 0.1,26,0,3 ]
            cmd += f'--transform {transform}[' + gradientStep + ',2,0,3] '
        elif transform == 'TimeVaryingVelocityField':
            # TimeVaryingVelocityField[gradientStep,numberOfTimeIndices,updateFieldVarianceInVoxelSpace,updateFieldTimeVariance,totalFieldVarianceInVoxelSpace,totalFieldTimeVariance]
            # Default values from antsMultivariateTemplateConstruction2.sh : [ 0.5,4,3,0,0,0 ]
            cmd += f'--transform {transform}[' + gradientStep + ',4,2,0,0,0] '
        elif transform == 'TimeVaryingBSplineVelocityField':
            # TimeVaryingBSplineVelocityField[gradientStep,velocityFieldMeshSize,<numberOfTimePointSamples=4>,<splineOrder=3>]
            # Default values from antsMultivariateTemplateConstruction2.sh : [ 0.5,12x12x12x2,4,3 ]
            cmd += f'--transform {transform}[' + gradientStep + ',10,0] '

        if use_seg is True:
            weighting_seg = 1.0 / len(label_range)
            for label in label_range:
                starting_lb_path = os.path.join(path_labels, f'tissue-t{starting_week}.00_label_{label}.nii.gz')
                current_lb_path = os.path.join(path_labels, f'tissue-t{week}.00_label_{label}.nii.gz')
                cmd += f'--metric MeanSquares[{current_lb_path}, {starting_lb_path},{weighting_seg}] '

        if use_intensity is True:
            week_intensity = os.path.join(path_intensity, f't2-t{week}.00.nii.gz')
            # changement d'intensité lié à la maturation -> CC/MI et non MeanSquare
            if loss == 'CC':
                cmd += f'--metric CC[{week_intensity},{path_intensity_starting},1,3] '
            if loss == 'MI':
                cmd += f'--metric MI[{week_intensity},{path_intensity_starting},1,32] '

        cmd += '--convergence [1000x500x500,1e-6,10] '
        cmd += '--shrink-factors 4x2x1 '
        cmd += '--smoothing-sigmas 2x1x0vox '
        cmd += '--verbose 1'

        print(cmd)
        os.system(cmd)

        ending_path = os.path.join(path_intensity, f't2-t{week}.00.nii.gz')

        # Apply the transformation
        cmd = f'antsApplyTransforms -d 3 -r {ending_path} -i {path_intensity_starting} -t {prefix}0Warp.nii.gz -o {prefix}_21_on_{week}_Warped.nii.gz '
        print(cmd)
        os.system(cmd)
        cmd = f"mv {prefix}0Warp.nii.gz {prefix}_21_on_{week}_Warp.nii.gz"
        os.system(cmd)

