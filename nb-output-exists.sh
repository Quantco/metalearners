for notebook in docs/examples/*.ipynb; do
    # Check the notebook for null execution counts
    if ! jq -e '.cells | map(select(.cell_type == "code") | .execution_count != null) | all' "$notebook" > /dev/null; then
        echo "Error: $notebook has code cells with null execution counts."
        exit 1
    fi
done

# If all notebooks are valid
echo "All notebooks have valid execution counts."
exit 0
