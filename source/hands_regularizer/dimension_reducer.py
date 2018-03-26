from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hands_bounding_utils.hands_locator_from_rgbd import *
import hands_bounding_utils.utils as u


dataset_path = pm.resources_path(os.path.join("hands_bounding_dataset", "network_test"))

images, heat_maps, depths = read_dataset_random(path=dataset_path, number=904)

# Make an instance of the Model
pca = PCA(1 - 1e-2)
scaler = StandardScaler()

flattened_images = np.reshape(images, newshape=(len(images), -1))

scaler.fit(flattened_images)
flattened_images = scaler.transform(flattened_images)

test = flattened_images[900:]
flattened_images = flattened_images[:900]

print(len(flattened_images[1]))
pca.fit(flattened_images)

approx_test = pca.inverse_transform(pca.transform(test))
approx_images = np.reshape(approx_test, newshape=(4, ) + np.shape(images)[1:])
approx_images = approx_images.clip(min=0, max=1)
print(len(pca.components_))

for i in range(4):
    u.showimage(images[i])
    u.showimage(approx_images[i])
