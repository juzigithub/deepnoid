# Set up our usual routines and configuration
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
# - set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib
import aneurysm_yonsei.utils as utils
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


moving_img = nib.load('d:\\nipype\\14.nii.gz')
template_img = nib.load('d:\\nipype\\15.nii.gz')

moving_data = moving_img.get_data()
moving_affine = moving_img.affine
template_data = template_img.get_data()
template_affine = template_img.affine

identity = np.eye(4)
affine_map = AffineMap(identity,
                       template_data.shape, template_affine,
                       moving_data.shape, moving_affine)
resampled = affine_map.transform(moving_data)
regtools.overlay_slices(template_data, resampled, None, 0,
                        "Template", "Moving")

regtools.overlay_slices(template_data, resampled, None, 1,
                        "Template", "Moving")

regtools.overlay_slices(template_data, resampled, None, 2,
                        "Template", "Moving")

# The mismatch metric
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

# The optimization strategy
level_iters = [10, 10, 5]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

transform = TranslationTransform3D()
params0 = None
translation = affreg.optimize(template_data, moving_data, transform, params0,
                              template_affine, moving_affine)

print(translation.affine)

transformed = translation.transform(moving_data)
# transformed = np.transpose(transformed, (1,0,2))
utils.save_array_as_nifty_volume(transformed, 'd:\\nipype\\11_transformed1.nii.gz')
print('saved')
regtools.overlay_slices(template_data, transformed, None, 0,
                        "Template", "Transformed")

regtools.overlay_slices(template_data, transformed, None, 1,
                        "Template", "Transformed")

regtools.overlay_slices(template_data, transformed, None, 2,
                        "Template", "Transformed")

transform = RigidTransform3D()
rigid = affreg.optimize(template_data, moving_data, transform, params0,
                        template_affine, moving_affine,
                        starting_affine=translation.affine)

print(rigid.affine)

transformed = rigid.transform(moving_data)

transformed = np.transpose(transformed, (1,0,2))
utils.save_array_as_nifty_volume(transformed, 'd:\\nipype\\11_transformed2.nii.gz')
print('saved')
regtools.overlay_slices(template_data, transformed, None, 0,
                        "Template", "Transformed")

regtools.overlay_slices(template_data, transformed, None, 1,
                        "Template", "Transformed")

regtools.overlay_slices(template_data, transformed, None, 2,
                        "Template", "Transformed")

transform = AffineTransform3D()
# Bump up the iterations to get an more exact fit
affreg.level_iters = [1000, 1000, 100]
affine = affreg.optimize(template_data, moving_data, transform, params0,
                         template_affine, moving_affine,
                         starting_affine=rigid.affine)

print(affine.affine)

transformed = affine.transform(moving_data)
transformed = np.transpose(transformed, (1,0,2))
utils.save_array_as_nifty_volume(transformed, 'd:\\nipype\\11_transformed3.nii.gz')
print('saved')

regtools.overlay_slices(template_data, transformed, None, 0,
                        "Template", "Transformed")

regtools.overlay_slices(template_data, transformed, None, 1,
                        "Template", "Transformed")

regtools.overlay_slices(template_data, transformed, None, 2,
                        "Template", "Transformed")