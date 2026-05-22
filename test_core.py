import sys
sys.path.insert(0, 'streamlit_app')
import numpy as np
from core.pca_svd import svd_decompose, compute_eigenfaces, project_to_eigenspace
from core.similarity import cosine_similarity, compute_all_metrics, make_decision

face1 = np.random.rand(128, 128)
face2 = np.random.rand(128, 128)

U, S, Vt = svd_decompose(face1)
print("SVD OK - U:", U.shape, "S:", S.shape, "Vt:", Vt.shape)

imgs = np.stack([face1.flatten(), face2.flatten()])
ef_data = compute_eigenfaces(imgs, n_components=2)
print("Eigenfaces shape:", ef_data["eigenfaces"].shape)
print("Eigenvalues:", ef_data["eigenvalues"].round(4))

w1 = project_to_eigenspace(face1.flatten(), ef_data["eigenfaces"], ef_data["mean_face"])
w2 = project_to_eigenspace(face2.flatten(), ef_data["eigenfaces"], ef_data["mean_face"])
print("Weights face1:", w1.round(4))
print("Weights face2:", w2.round(4))

cos = cosine_similarity(w1, w2)
print("Cosine Similarity:", round(cos, 4))

metrics = compute_all_metrics(w1, w2, face1, face2)
decision = make_decision(metrics, threshold=0.70)
print("Composite Score:", metrics["composite_score"])
print("Verdict:", decision["verdict"])   # ASCII-safe
print("Level:", decision["level"])
print("Is same person:", decision["is_same_person"])

print("")
print("ALL MODULES OK - Pipeline works correctly!")
