from os.path import join as opj
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.freesurfer import FSCommand, MRIConvert, BBRegister
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.utility import IdentityInterface, Merge
from nipype.interfaces.io import SelectFiles, DataSink, FreeSurferSource
from nipype.pipeline.engine import Workflow, Node, MapNode
# from nipype.interfaces.fsl import Info

# template = Info.standard_image('C:\\Users\\sunki\\PycharmProjects\\deepnoid\\aneurysm_yonsei\\nifti\\30.nii.gz')
# template = 'd:\\nipype\\11.nii.gz'
template = '/home/mspark/project/aneurysm/nipype/11.nii.gz'



antsreg = Node(Registration(args='--float',
                            collapse_output_transforms=True,
                            fixed_image=template,
                            initial_moving_transform_com=True,
                            num_threads=1,
                            output_inverse_warped_image=True,
                            output_warped_image=True,
                            sigma_units=['vox']*3,
                            transforms=['Rigid', 'Affine', 'SyN'],
                            terminal_output='file',
                            winsorize_lower_quantile=0.005,
                            winsorize_upper_quantile=0.995,
                            convergence_threshold=[1e-06],
                            convergence_window_size=[10],
                            metric=['MI', 'MI', 'CC'],
                            metric_weight=[1.0]*3,
                            number_of_iterations=[[1000, 500, 250, 100],
                                                  [1000, 500, 250, 100],
                                                  [100, 70, 50, 20]],
                            radius_or_number_of_bins=[32, 32, 4],
                            sampling_percentage=[0.25, 0.25, 1],
                            sampling_strategy=['Regular',
                                               'Regular',
                                               'None'],
                            shrink_factors=[[8, 4, 2, 1]]*3,
                            smoothing_sigmas=[[3, 2, 1, 0]]*3,
                            transform_parameters=[(0.1,),
                                                  (0.1,),
                                                  (0.1, 3.0, 0.0)],
                            use_histogram_matching=True,
                            write_composite_transform=True),
               name='antsreg')

# Apply Transformation - applies the normalization matrix to contrast images
apply2con = MapNode(ApplyTransforms(args='--float',
                                    input_image_type=3,
                                    interpolation='Linear',
                                    invert_transform_flags=[False],
                                    num_threads=1,
                                    reference_image=template,
                                    terminal_output='file'),
                    name='apply2con', iterfield=['input_image'])

# Apply Transformation - applies the normalization matrix to the mean image
apply2mean = Node(ApplyTransforms(args='--float',
                                  input_image_type=3,
                                  interpolation='Linear',
                                  invert_transform_flags=[False],
                                  num_threads=1,
                                  reference_image=template,
                                  terminal_output='file'),
                  name='apply2mean')

# Coregister the median to the surface
# bbregister = Node(BBRegister(init='fsl',
#                              contrast_type='t2',
#                              out_fsl_file=True),
#                   name='bbregister')

# Convert the BBRegister transformation to ANTS ITK format
# convert2itk = Node(C3dAffineTool(fsl2ras=True,
#                                  itk_transform=True),
#                    name='convert2itk')


# Concatenate BBRegister's and ANTS' transforms into a list
# merge = Node(Merge(2), iterfield=['in2'], name='mergexfm')

# Transform the contrast images. First to anatomical and then to the target
warpall = MapNode(ApplyTransforms(args='--float',
                                  input_image_type=3,
                                  interpolation='Linear',
                                  invert_transform_flags=[False, False],
                                  num_threads=1,
                                  reference_image=template,
                                  terminal_output='file'),
                  name='warpall', iterfield=['input_image'])

# Transform the mean image. First to anatomical and then to the target
warpmean = Node(ApplyTransforms(args='--float',
                                input_image_type=3,
                                interpolation='Linear',
                                invert_transform_flags=[False, False],
                                num_threads=1,
                                reference_image=template,
                                terminal_output='file'),
                name='warpmean')


# Initiation of the ANTS normalization workflow
normflow = Workflow(name='normflow')
# normflow.base_dir = opj(experiment_dir, working_dir)
normflow.base_dir = '/home/mspark/project/aneurysm/nipype/'



# Connect up ANTS normalization components
normflow.connect([(antsreg, apply2con, [('composite_transform', 'transforms')]),
                  (antsreg, apply2mean, [('composite_transform',
                                          'transforms')])
                  ])



# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
# infosource.iterables = [('subject_id', subject_list)]
infosource.iterables = [('subject_id', [11,12,13,14,15,16])]


# SelectFiles - to grab the data (alternativ to DataGrabber)
# anat_file = opj('freesurfer', '{subject_id}', 'mri/brain.mgz')
# func_file = opj(input_dir_1st, 'contrasts', '{subject_id}',
#                 '_mriconvert*/*_out.nii.gz')
# func_orig_file = opj(input_dir_1st, 'contrasts', '{subject_id}', '[ce]*.nii')
# mean_file = opj(input_dir_1st, 'preprocout', '{subject_id}', 'mean*.nii')


# anat_file = opj('freesurfer', '{subject_id}', 'mri/brain.mgz')
# func_file = '/home/mspark/project/aneurysm/nipype/{}_out.nii.gz'.format('{subject_id}')
func_orig_file = '/home/mspark/project/aneurysm/nipype/{}.nii.gz'.format('{subject_id}')
# mean_file = opj(input_dir_1st, 'preprocout', '{subject_id}', 'mean*.nii')


# templates = {'anat': anat_file,
#              'func': func_file,
#              'func_orig': func_orig_file,
#              'mean': mean_file,
#              }
templates = {'anat' : '/home/mspark/project/aneurysm/nipype/11.nii.gz',

             'func_orig': func_orig_file,

             }




selectfiles = Node(SelectFiles(templates,
                               base_directory='/home/mspark/project/aneurysm/nipype/'),
                   name="selectfiles")

# Datasink - creates output folder for important outputs
# datasink = Node(DataSink(base_directory=experiment_dir,
#                          container=output_dir),
#                 name="datasink")

datasink = Node(DataSink(base_directory='/home/mspark/project/aneurysm/nipype/',
                         container='/home/mspark/project/aneurysm/nipypeout/'),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_subject_id_', ''),
                 ('_apply2con', 'apply2con'),
                 ('_warpall', 'warpall')]
datasink.inputs.substitutions = substitutions



# Connect SelectFiles and DataSink to the workflow
normflow.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                  (selectfiles, apply2con, [('func', 'input_image')]),
                  (selectfiles, apply2mean, [('mean', 'input_image')]),
                  (selectfiles, antsreg, [('anat', 'moving_image')]),
                  (antsreg, datasink, [('warped_image',
                                        'antsreg.@warped_image'),
                                       ('inverse_warped_image',
                                        'antsreg.@inverse_warped_image'),
                                       ('composite_transform',
                                        'antsreg.@transform'),
                                       ('inverse_composite_transform',
                                        'antsreg.@inverse_transform')]),
                  (apply2con, datasink, [('output_image',
                                          'warp_partial.@con')]),
                  (apply2mean, datasink, [('output_image',
                                          'warp_partial.@mean')]),
                  ])

normflow.write_graph(graph2use='colored')
normflow.run('MultiProc', plugin_args={'n_procs': 8})