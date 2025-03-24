from graphviz import Digraph

def create_pipeline_diagram():
    dot = Digraph(format='png')
    dot.attr(bgcolor='white')

    # Define colors
    colors = {
        "Step 1": "#1E88E5",  # Blue
        "Step 2": "#43A047",  # Green
        "Step 3": "#FDD835",  # Yellow
        "Step 4": "#FB8C00",  # Orange
        "Step 5": "#D81B60"   # Pink
    }

    # Step 1: Malaria & Climate Dataset with Images
    dot.node("DB1", label="Malaria Dataset", image="malaria_db.png", shape="none", width="0.5")
    dot.node("DB2", label="Climate Dataset", image="climate_db.png", shape="none", width="0.5")

    dot.node("Step 1", "Data Collection\n- Load Malaria Data\n- Load Climate Data\n- Align Data",
             style="filled", fillcolor=colors["Step 1"], shape="box", fontsize="14", fontcolor="white")

    # Step 2: Preprocessing
    dot.node("Step 2", "Preprocessing\n- Handle Missing Data\n- Feature Scaling\n- Train-Test Split",
             style="filled", fillcolor=colors["Step 2"], shape="box", fontsize="14", fontcolor="white")

    # Step 3: Feature Engineering
    dot.node("Step 3", "Feature Engineering\n- Extract Patterns\n- Create New Features",
             style="filled", fillcolor=colors["Step 3"], shape="box", fontsize="14", fontcolor="black")

    # Step 4: Hyperparameter Optimization
    dot.node("Step 4", "Hyperparameter Optimization\n- Grid Search\n- Random Search",
             style="filled", fillcolor=colors["Step 4"], shape="box", fontsize="14", fontcolor="white")

    # Step 5: Model Training & Evaluation
    dot.node("Step 5", "Model Training & Evaluation\n- Train Model\n- Evaluate Results",
             style="filled", fillcolor=colors["Step 5"], shape="box", fontsize="14", fontcolor="white")

    # Connect database images to Step 1
    dot.edge("DB1", "Step 1")
    dot.edge("DB2", "Step 1")

    # Connect Steps
    dot.edge("Step 1", "Step 2")
    dot.edge("Step 2", "Step 3")
    dot.edge("Step 3", "Step 4")
    dot.edge("Step 4", "Step 5")

    # Save and render the diagram
    dot.render("malaria_pipeline", format="png", cleanup=True)
    print("Pipeline diagram saved as 'malaria_pipeline.png' ✅")

# Run the function
create_pipeline_diagram()
