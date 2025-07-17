#!/bin/bash

# Qdrant Manager Script
# Manages Qdrant Docker container for ColBERT evaluation

set -e

CONTAINER_NAME="qdrant_colbert"
IMAGE_NAME="qdrant/qdrant:latest"
PORT="6333"
COLLECTION_NAME="colbert_scidocs"
DATA_DIR="./qdrant_data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if container exists
container_exists() {
    docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to wait for Qdrant to be ready
wait_for_qdrant() {
    print_status "Waiting for Qdrant to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            print_success "Qdrant is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Qdrant failed to start within expected time"
    return 1
}

# Function to create collection
create_collection() {
    print_status "Creating collection: ${COLLECTION_NAME}"
    
    # Collection configuration for ColBERT (128-dim vectors, dot product)
    curl -X PUT "http://localhost:${PORT}/collections/${COLLECTION_NAME}" \
        -H "Content-Type: application/json" \
        -d '{
            "vectors": {
                "size": 128,
                "distance": "Dot"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        }' > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Collection '${COLLECTION_NAME}' created successfully"
    else
        print_warning "Collection might already exist or creation failed"
    fi
}

# Function to delete collection
delete_collection() {
    print_status "Deleting collection: ${COLLECTION_NAME}"
    
    curl -X DELETE "http://localhost:${PORT}/collections/${COLLECTION_NAME}" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Collection '${COLLECTION_NAME}' deleted successfully"
    else
        print_warning "Collection might not exist or deletion failed"
    fi
}

# Function to start Qdrant
start_qdrant() {
    local clear_data=false
    
    # Check for --clear flag
    if [ "$1" = "--clear" ]; then
        clear_data=true
        print_warning "Clear flag detected - will remove existing data"
    fi
    
    check_docker
    
    # If clear flag is set, stop and remove container and data
    if [ "$clear_data" = true ]; then
        print_status "Clearing existing Qdrant data..."
        
        if container_running; then
            print_status "Stopping existing container..."
            docker stop $CONTAINER_NAME > /dev/null 2>&1
        fi
        
        if container_exists; then
            print_status "Removing existing container..."
            docker rm $CONTAINER_NAME > /dev/null 2>&1
        fi
        
        if [ -d "$DATA_DIR" ]; then
            print_status "Removing data directory..."
            rm -rf "$DATA_DIR"
        fi
    fi
    
    # Check if container is already running
    if container_running; then
        print_success "Qdrant container is already running"
        return 0
    fi
    
    # Check if container exists but is stopped
    if container_exists; then
        print_status "Starting existing Qdrant container..."
        docker start $CONTAINER_NAME > /dev/null 2>&1
    else
        print_status "Creating and starting new Qdrant container..."
        
        # Create data directory
        mkdir -p "$DATA_DIR"
        
        # Run Qdrant container
        docker run -d \
            --name $CONTAINER_NAME \
            -p $PORT:6333 \
            -p 6334:6334 \
            -v "$(pwd)/$DATA_DIR:/qdrant/storage" \
            $IMAGE_NAME > /dev/null 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Qdrant container started successfully"
        
        # Wait for Qdrant to be ready
        if wait_for_qdrant; then
            # Create collection
            create_collection
            print_success "Qdrant is ready for ColBERT evaluation!"
            print_status "Access Qdrant dashboard at: http://localhost:${PORT}/dashboard"
        else
            print_error "Failed to start Qdrant properly"
            exit 1
        fi
    else
        print_error "Failed to start Qdrant container"
        exit 1
    fi
}

# Function to stop Qdrant
stop_qdrant() {
    check_docker
    
    if container_running; then
        print_status "Stopping Qdrant container..."
        docker stop $CONTAINER_NAME > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            print_success "Qdrant container stopped successfully"
        else
            print_error "Failed to stop Qdrant container"
            exit 1
        fi
    else
        print_warning "Qdrant container is not running"
    fi
}

# Function to show status
show_status() {
    check_docker
    
    echo "=== Qdrant Status ==="
    echo "Container name: $CONTAINER_NAME"
    echo "Port: $PORT"
    echo "Collection: $COLLECTION_NAME"
    echo "Data directory: $DATA_DIR"
    echo ""
    
    if container_running; then
        print_success "Status: RUNNING"
        echo "Dashboard: http://localhost:${PORT}/dashboard"
        echo ""
        
        # Check if collection exists
        if curl -s "http://localhost:${PORT}/collections/${COLLECTION_NAME}" > /dev/null 2>&1; then
            print_success "Collection '${COLLECTION_NAME}' exists"
            
            # Get collection info
            collection_info=$(curl -s "http://localhost:${PORT}/collections/${COLLECTION_NAME}")
            points_count=$(echo "$collection_info" | grep -o '"points_count":[0-9]*' | cut -d':' -f2)
            if [ -n "$points_count" ]; then
                echo "Points in collection: $points_count"
            fi
        else
            print_warning "Collection '${COLLECTION_NAME}' does not exist"
        fi
    elif container_exists; then
        print_warning "Status: STOPPED"
    else
        print_warning "Status: NOT CREATED"
    fi
}

# Function to show logs
show_logs() {
    check_docker
    
    if container_exists; then
        print_status "Showing Qdrant logs (last 50 lines)..."
        docker logs --tail 50 $CONTAINER_NAME
    else
        print_error "Qdrant container does not exist"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Qdrant Manager for ColBERT Evaluation"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [--clear]    Start Qdrant container and create collection"
    echo "                     --clear: Remove existing data and start fresh"
    echo "  stop               Stop Qdrant container"
    echo "  status             Show Qdrant status and collection info"
    echo "  logs               Show Qdrant container logs"
    echo "  restart [--clear]  Restart Qdrant (stop + start)"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start Qdrant with existing data"
    echo "  $0 start --clear   # Start Qdrant with fresh data"
    echo "  $0 stop            # Stop Qdrant"
    echo "  $0 status          # Check status"
    echo "  $0 restart --clear # Restart with fresh data"
}

# Main script logic
case "$1" in
    start)
        start_qdrant "$2"
        ;;
    stop)
        stop_qdrant
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    restart)
        stop_qdrant
        sleep 2
        start_qdrant "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
