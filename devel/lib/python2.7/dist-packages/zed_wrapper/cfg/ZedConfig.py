## *********************************************************
##
## File autogenerated for the zed_wrapper package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'groups': [{'groups': [], 'id': 1, 'srcline': 124, 'name': 'general', 'parent': 0, 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'parentname': 'Default', 'state': True, 'lower': 'general', 'cstate': 'true', 'parentclass': 'DEFAULT', 'upper': 'GENERAL', 'class': 'DEFAULT::GENERAL', 'parameters': [{'min': 0.1, 'edit_method': '', 'name': 'mat_resize_factor', 'srcline': 9, 'ctype': 'double', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 0, 'max': 1.0, 'cconsttype': 'const double', 'type': 'double', 'description': 'Image/Measures resize factor', 'default': 1.0}], 'type': '', 'field': 'DEFAULT::general'}, {'groups': [], 'id': 2, 'srcline': 124, 'name': 'depth', 'parent': 0, 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'parentname': 'Default', 'state': True, 'lower': 'depth', 'cstate': 'true', 'parentclass': 'DEFAULT', 'upper': 'DEPTH', 'class': 'DEFAULT::DEPTH', 'parameters': [{'min': 1, 'edit_method': '', 'name': 'confidence', 'srcline': 12, 'ctype': 'int', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 1, 'max': 100, 'cconsttype': 'const int', 'type': 'int', 'description': 'Confidence threshold, the lower the better', 'default': 100}, {'min': 0.5, 'edit_method': '', 'name': 'max_depth', 'srcline': 13, 'ctype': 'double', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 2, 'max': 20.0, 'cconsttype': 'const double', 'type': 'double', 'description': 'Maximum Depth Range', 'default': 3.5}, {'min': 0.1, 'edit_method': '', 'name': 'point_cloud_freq', 'srcline': 14, 'ctype': 'double', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 3, 'max': 60.0, 'cconsttype': 'const double', 'type': 'double', 'description': 'Point cloud frequency', 'default': 15.0}], 'type': '', 'field': 'DEFAULT::depth'}, {'groups': [], 'id': 3, 'srcline': 124, 'name': 'video', 'parent': 0, 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'parentname': 'Default', 'state': True, 'lower': 'video', 'cstate': 'true', 'parentclass': 'DEFAULT', 'upper': 'VIDEO', 'class': 'DEFAULT::VIDEO', 'parameters': [{'min': False, 'edit_method': '', 'name': 'auto_exposure', 'srcline': 17, 'ctype': 'bool', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 4, 'max': True, 'cconsttype': 'const bool', 'type': 'bool', 'description': 'Enable/Disable auto control of exposure and gain', 'default': True}, {'min': 0, 'edit_method': '', 'name': 'gain', 'srcline': 18, 'ctype': 'int', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 5, 'max': 100, 'cconsttype': 'const int', 'type': 'int', 'description': 'Gain value when manual controlled', 'default': 100}, {'min': 0, 'edit_method': '', 'name': 'exposure', 'srcline': 19, 'ctype': 'int', 'srcfile': '/home/klab/catkin_ws/src/zed-ros-wrapper/zed_wrapper/cfg/Zed.cfg', 'level': 6, 'max': 100, 'cconsttype': 'const int', 'type': 'int', 'description': 'Exposure value when manual controlled', 'default': 100}], 'type': '', 'field': 'DEFAULT::video'}], 'id': 0, 'srcline': 245, 'name': 'Default', 'parent': 0, 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'parentname': 'Default', 'state': True, 'lower': 'groups', 'cstate': 'true', 'parentclass': '', 'upper': 'DEFAULT', 'class': 'DEFAULT', 'parameters': [], 'type': '', 'field': 'default'}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

