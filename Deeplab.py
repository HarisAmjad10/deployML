import tensorflow as tf

class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		#"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		graph_def = None
		# Extract frozen graph from tar archive.
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()

		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		  image: A PIL.Image object, raw input image.

		Returns:
		  resized_image: RGB image resized from original input image.
		  seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

def create_pascal_label_colormap():
	"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

	Returns:
	A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
		  colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3

	return colormap

def label_to_color_image(label):
	"""Adds color defined by the dataset colormap to the label.

	Args:
	label: A 2D array with integer type, storing the segmentation label.

	Returns:
	result: A 2D array with floating type. The element of the array
	  is the color indexed by the corresponding element in the input label
	  to the PASCAL color map.

	Raises:
	ValueError: If label is not of rank 2 or its value is larger than color
	  map maximum entry.
	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_pascal_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]





def preprocess_image(img_path, json_path=None):
    img = img_path#io.imread(img_path)
    print("$$$$$$$",img.shape)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != 224:
#            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(224) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               224)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, height, json_path=None):
#    renderer = vis_util.SMPLRenderer(face_path='src/tf_smpl/smpl_faces.npy')
    sess = tf.Session()
    model = RunModel(sess=sess)
#    cv2.imshow('input image for measurement extraction',img_path)
#    cv2.waitKey(0)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    all_measurements = extract_measurements.extract_measurements(height,verts[0])
    
    return all_measurements

