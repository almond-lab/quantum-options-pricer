# GKE Deployment — Quantum Options Pricer

## Prerequisites

### 1. GKE cluster with T4 GPU node pool

```bash
# Create cluster (if not exists)
gcloud container clusters create quantum-pricer-cluster \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type n1-standard-4

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster quantum-pricer-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 1
```

### 2. NVIDIA device plugin (one-time per cluster)

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-device-plugin.yaml
```

Verify GPU is visible:
```bash
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

### 3. Artifact Registry + push image

```bash
export PROJECT=your-gcp-project-id
export REGION=us

# Create repo (once)
gcloud artifacts repositories create quantum-pricer \
  --repository-format=docker \
  --location=$REGION

# Auth Docker
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push
docker build \
  --build-arg CUDA_ARCH=75 \
  -t ${REGION}-docker.pkg.dev/${PROJECT}/quantum-pricer/api:latest .

docker push ${REGION}-docker.pkg.dev/${PROJECT}/quantum-pricer/api:latest
```

Then update the image reference in `deployment.yaml`:
```yaml
image: us-docker.pkg.dev/YOUR_PROJECT/quantum-pricer/api:latest
```

---

## Deploy

```bash
# 1. Namespace
kubectl apply -f k8s/namespace.yaml

# 2. Secret (from your .env file)
kubectl -n quantum-pricer create secret generic quantum-pricer-secret \
  --from-env-file=.env \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Watch rollout
kubectl -n quantum-pricer rollout status deployment/quantum-pricer
```

---

## Verify

```bash
# Pod is Running, GPU allocated
kubectl -n quantum-pricer get pods -o wide

# Health endpoint
kubectl -n quantum-pricer port-forward svc/quantum-pricer 8000:80
curl http://localhost:8000/health
# → {"backend":"gpu","sampler_type":"AerSamplerV2","gpu_available":true,...}

# Logs
kubectl -n quantum-pricer logs -f deployment/quantum-pricer
```

---

## External access

**Option A — Quick test (LoadBalancer):**
Change `spec.type` in `service.yaml` to `LoadBalancer`, then:
```bash
kubectl -n quantum-pricer get svc quantum-pricer
# → EXTERNAL-IP when provisioned (~60s)
curl http://<EXTERNAL-IP>/health
```

**Option B — Production (Ingress + TLS):**
Use GKE Ingress or Gateway API with a managed certificate. The service is
already ClusterIP-ready; attach an Ingress pointing at port 80.

---

## Switching from T4 to L4 (Ada Lovelace)

1. Change node pool accelerator to `nvidia-l4`
2. In `deployment.yaml` change `nodeSelector`:
   ```yaml
   cloud.google.com/gke-accelerator: nvidia-l4
   ```
3. Rebuild the image with `--build-arg CUDA_ARCH=89`

---

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pod stuck `Pending` | No GPU node available / taint not tolerated | Check node pool size; verify toleration in deployment.yaml |
| `nvidia.com/gpu: 0` in `kubectl describe` | Device plugin not installed | Apply NVIDIA device plugin daemonset |
| `CUDA error: no kernel image` | Wrong `CUDA_ARCH` for GPU | Rebuild with correct `--build-arg CUDA_ARCH` (75=T4, 89=L4) |
| `ImportError: qiskit_aer_gpu` | Image built without GPU support | Ensure builder stage uses `cudnn9-devel` base |
| Secret not found | Secret in wrong namespace | Create secret in `quantum-pricer` namespace |
