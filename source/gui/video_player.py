# TODO
# Implement a video player to merge frame data with label data of a certain video
# Suggested classes to use:
# - in gui.model_drawer.py: ModelDrawer
#   instanciate:        md = ModelDrawer()
#   set full canvas:    md.set_target_area(canvas)
#   when needed draw:   md.set_joints(joints)
#   --> see the doc of the class and the two methods
# - in image_loader.hand_io.py: load()
#   --> see the doc of the function
#
# about reading video data and interpolating missing labels
# a standalone module may be done
# because it is needed also for training
# - in hand_data_management.video_loader: function load_labeled_video()
#   --> see the doc of the function
