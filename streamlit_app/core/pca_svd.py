import numpy as np
from typing import Tuple, Optional, Dict

def svd_decompose(image_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_matrix.ndim == 1:
        image_matrix = image_matrix.reshape(1, -1)
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    return U, S, Vt


def get_singular_values_info(S: np.ndarray) -> dict:
    total_energy = np.sum(S ** 2)
    explained_variance = (S ** 2) / total_energy * 100
    cumulative_variance = np.cumsum(explained_variance)
    return {
        "singular_values": S,
        "explained_variance_pct": explained_variance,
        "cumulative_variance_pct": cumulative_variance,
        "total_energy": total_energy,
    }

def load_olivetti_dataset() -> Dict:
    from sklearn.datasets import fetch_olivetti_faces
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)

    return {
        "images"      : dataset.data,
        "images_2d"   : dataset.images,
        "targets"     : dataset.target,
        "n_samples"   : dataset.data.shape[0],
        "n_people"    : len(np.unique(dataset.target)),
        "image_shape" : (64, 64),
        "pixel_size"  : 4096,
        "source"      : "Olivetti Faces (AT&T -- 400 foto, 40 orang)",
        "description" : (
            "Dataset Olivetti Faces dari AT&T Laboratories Cambridge. "
            "400 foto wajah dari 40 orang berbeda (10 foto per orang). "
            "Ukuran: 64x64 piksel, grayscale, float [0,1]. "
            "Digunakan sebagai training set untuk membangun eigenspace."
        ),
    }


def load_lfw_dataset(min_faces: int = 20, resize: float = 0.4) -> Optional[Dict]:
    try:
        from sklearn.datasets import fetch_lfw_people
        dataset = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
        h, w = dataset.images.shape[1], dataset.images.shape[2]
        n = dataset.data.shape[0]
        return {
            "images"       : dataset.data,
            "images_2d"    : dataset.images,
            "targets"      : dataset.target,
            "target_names" : dataset.target_names,
            "n_samples"    : n,
            "n_people"     : len(dataset.target_names),
            "image_shape"  : (h, w),
            "pixel_size"   : h * w,
            "source"       : "LFW (Labeled Faces in the Wild -- {} foto)".format(n),
            "description"  : (
                "Dataset LFW berisi {} foto wajah dari {} orang terkenal. "
                "Ukuran: {}x{} piksel.".format(n, len(dataset.target_names), h, w)
            ),
        }
    except Exception:
        return None

def compute_mean_face(images: np.ndarray) -> np.ndarray:
    return np.mean(images, axis=0)


def compute_covariance_matrix(images_centered: np.ndarray) -> np.ndarray:
    n = images_centered.shape[0]
    C = (images_centered.T @ images_centered) / n
    return C


def compute_eigenfaces(
    images: np.ndarray,
    n_components: int = 50,
    use_svd: bool = True,
) -> dict:
    if images.ndim == 3:
        n = images.shape[0]
        images = images.reshape(n, -1)

    images = images.astype(float)
    mean_face = compute_mean_face(images)
    images_centered = images - mean_face

    n_components = min(n_components, images.shape[0], images.shape[1])

    if use_svd:
        U, S, Vt = np.linalg.svd(images_centered, full_matrices=False)
        eigenfaces  = Vt[:n_components]
        eigenvalues = (S[:n_components] ** 2) / images.shape[0]
        ev_ratio    = (S ** 2) / np.sum(S ** 2)
        singular_values = S
    else:
        C = compute_covariance_matrix(images_centered)
        eigenvalues_all, eigenvectors_all = np.linalg.eig(C)
        idx = np.argsort(eigenvalues_all)[::-1]
        eigenvalues_all  = eigenvalues_all[idx].real
        eigenvectors_all = eigenvectors_all[:, idx].real
        eigenfaces  = eigenvectors_all[:, :n_components].T
        eigenvalues = eigenvalues_all[:n_components]
        ev_ratio    = eigenvalues_all / np.sum(eigenvalues_all)
        singular_values = None

    return {
        "eigenfaces"              : eigenfaces,
        "eigenvalues"             : eigenvalues,
        "mean_face"               : mean_face,
        "explained_variance_ratio": ev_ratio[:n_components],
        "n_components"            : n_components,
        "singular_values"         : singular_values,
        "n_training_images"       : images.shape[0],
    }

def build_eigenspace_from_dataset(
    dataset_images: np.ndarray,
    n_components: int = 50,
    target_size: Tuple[int, int] = (64, 64),
) -> dict:
    eigenface_data = compute_eigenfaces(dataset_images, n_components=n_components)

    cumvar = np.cumsum(eigenface_data["explained_variance_ratio"])
    k_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    k_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    total_var = float(np.sum(eigenface_data["explained_variance_ratio"]))

    eigenface_data.update({
        "target_size"            : target_size,
        "k_for_95pct_variance"   : k_95,
        "k_for_99pct_variance"   : k_99,
        "total_variance_captured": total_var,
        "dataset_size"           : dataset_images.shape[0],
    })
    return eigenface_data

def project_to_eigenspace(
    face: np.ndarray,
    eigenfaces: np.ndarray,
    mean_face: np.ndarray,
) -> np.ndarray:
    if face.ndim > 1:
        face = face.flatten()
    face_centered = face.astype(float) - mean_face
    weights = eigenfaces @ face_centered
    return weights


def reconstruct_from_eigenspace(
    weights: np.ndarray,
    eigenfaces: np.ndarray,
    mean_face: np.ndarray,
) -> np.ndarray:
    return mean_face + weights @ eigenfaces


def resize_face_for_eigenspace(
    face: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    import cv2
    resized = cv2.resize(
        face.astype(np.float32),
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float64)

def analyze_two_faces(
    face1: np.ndarray,
    face2: np.ndarray,
    n_components: int = 2,
) -> dict:
    f1 = face1.flatten().astype(float)
    f2 = face2.flatten().astype(float)
    images_stack = np.stack([f1, f2], axis=0)
    n_comp = min(n_components, 2)
    eigenface_data = compute_eigenfaces(images_stack, n_components=n_comp)
    w1 = project_to_eigenspace(f1, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    w2 = project_to_eigenspace(f2, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    r1 = reconstruct_from_eigenspace(w1, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    r2 = reconstruct_from_eigenspace(w2, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    U1, S1, Vt1 = svd_decompose(face1.astype(float))
    U2, S2, Vt2 = svd_decompose(face2.astype(float))
    return {
        "face1_flat": f1, "face2_flat": f2,
        "eigenface_data": eigenface_data,
        "weights_face1": w1, "weights_face2": w2,
        "reconstructed_face1": r1, "reconstructed_face2": r2,
        "svd_face1": {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2": {"U": U2, "S": S2, "Vt": Vt2},
    }


def analyze_two_faces_with_dataset(
    face1: np.ndarray,
    face2: np.ndarray,
    eigenspace: dict,
) -> dict:
    eigenfaces  = eigenspace["eigenfaces"]
    mean_face   = eigenspace["mean_face"]
    target_size = eigenspace.get("target_size", (64, 64))

    face1_r = resize_face_for_eigenspace(face1, target_size)
    face2_r = resize_face_for_eigenspace(face2, target_size)

    f1 = face1_r.flatten()
    f2 = face2_r.flatten()

    w1 = project_to_eigenspace(f1, eigenfaces, mean_face)
    w2 = project_to_eigenspace(f2, eigenfaces, mean_face)

    r1 = reconstruct_from_eigenspace(w1, eigenfaces, mean_face).reshape(target_size)
    r2 = reconstruct_from_eigenspace(w2, eigenfaces, mean_face).reshape(target_size)

    U1, S1, Vt1 = svd_decompose(face1_r)
    U2, S2, Vt2 = svd_decompose(face2_r)

    return {
        "face1_resized"      : face1_r,
        "face2_resized"      : face2_r,
        "face1_flat"         : f1,
        "face2_flat"         : f2,
        "weights_face1"      : w1,
        "weights_face2"      : w2,
        "reconstructed_face1": r1,
        "reconstructed_face2": r2,
        "svd_face1"          : {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2"          : {"U": U2, "S": S2, "Vt": Vt2},
        "n_components_used"  : len(w1),
        "eigenspace_info"    : {
            "dataset_size"           : eigenspace.get("dataset_size", "?"),
            "n_components"           : eigenspace.get("n_components", "?"),
            "total_variance_captured": eigenspace.get("total_variance_captured", 0),
            "k_for_95pct"            : eigenspace.get("k_for_95pct_variance", "?"),
        },
    }
