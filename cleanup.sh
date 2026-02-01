#!/bin/bash
# cleanup.sh - Complete cleanup of ML Pipeline Kubernetes resources

set -e

NAMESPACE="ml-pipeline"

echo "=================================================="
echo " Cleaning up ML Pipeline Resources"
echo "=================================================="

# 1. Stop Port Forwarding
echo "[1/4] Stopping port-forwarding processes..."
pkill -f "kubectl port-forward" || true

# 2. Delete Jobs and Deployments
echo "[2/4] Deleting Jobs and Deployments in namespace '$NAMESPACE'..."
kubectl delete jobs --all -n $NAMESPACE --ignore-not-found
kubectl delete deployments --all -n $NAMESPACE --ignore-not-found
kubectl delete services --all -n $NAMESPACE --ignore-not-found

# 3. Optional: Remove Namespace
# echo "[3/4] Removing namespace '$NAMESPACE'..."
# kubectl delete namespace $NAMESPACE --ignore-not-found

# 4. Cleanup MLflow and Local Temporary Manifests
echo "[4/4] Cleaning up MLflow and generated files..."
kubectl delete deployment mlflow-server -n $NAMESPACE --ignore-not-found || true
kubectl delete service mlflow-service -n $NAMESPACE --ignore-not-found || true
rm -rf k8s/generated/

echo "=================================================="
echo " Cleanup Complete!"
echo "=================================================="
