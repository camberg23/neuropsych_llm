import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
import io
import random

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets['API_KEY'])

def get_embeddings(text):
    # Pseudocode for API call - replace with actual call
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

# Add or delete agent block
def add_agent_block():
    last_block_settings = st.session_state.agent_blocks[-1] if st.session_state.agent_blocks else {
        'system_message': '',
        'temperature': 0.5,
        'presence_penalty': 0,
        'logprobs': False,
        'model': 'gpt-4-1106-preview',
    }
    st.session_state.agent_blocks.append(last_block_settings.copy())

def delete_agent_block(index):
    if len(st.session_state.agent_blocks) > 1:
        del st.session_state.agent_blocks[index]
    else:
        st.error("You must have at least one agent block.")

# Initialize session state for agent blocks
if 'agent_blocks' not in st.session_state:
    st.session_state.agent_blocks = []
    add_agent_block()

if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None

if 'trial_seeds' not in st.session_state:
    st.session_state.trial_seeds = None

# Model selection and deletion button for each agent block
for index, block in enumerate(st.session_state.agent_blocks):
    with st.expander(f"Agent Block {index + 1}", expanded=True):
        block['model'] = st.selectbox(f'Model {index + 1}', ['gpt-4-1106-preview', 'gpt-3.5-turbo-0125'], index=0 if 'model' not in block else ['gpt-4-1106-preview', 'gpt-3.5-turbo-0125'].index(block['model']))
        block['system_message'] = st.text_area(f'System message {index + 1}', block['system_message'], height=100)
        block['temperature'] = st.number_input(f'Temperature {index + 1}', value=block['temperature'])
        block['presence_penalty'] = st.number_input(f'Presence Penalty {index + 1}', value=block['presence_penalty'])
        block['logprobs'] = st.checkbox(f'Logprobs {index + 1}', value=block['logprobs'])
        if st.button(f'Delete Block {index + 1}'):
            delete_agent_block(index)

# Add agent block button
if st.button('Add Agent Block'):
    add_agent_block()

# Configuration for trials and judge
num_trials = st.number_input('Number of Trials', min_value=1, value=1)
trial_seeds = [random.randint(0, 100000) for _ in range(num_trials)]
initialize_judge = st.checkbox('Initialize a Judge?')
judge_criteria = st.text_area("Judge's Criteria", 'Which output is more prosocial?') if initialize_judge else ''

# Before 'Run Experiment' button
analysis_options = st.multiselect(
    "Select analyses to perform:",
    ["qualitative analysis of results", "differentiability analyses", "results visualization"],
    []
)
st.session_state['selected_analyses'] = analysis_options

# Run Experiment button with new logic
if st.button('Run Experiment'):
    with st.spinner('Running experiment...'):
        experiment_results = []
        # Inside 'Run Experiment' button functionality
        for seed in trial_seeds:
            trial_result = {'responses': [], 'seed': seed}
            for block in st.session_state.agent_blocks:
                response = client.chat.completions.create(
                    model=block['model'],
                    messages=[{"role": "system", "content": block['system_message']}],
                    max_tokens=4096,
                    temperature=block['temperature'],
                    presence_penalty=block['presence_penalty'],
                    logprobs=block['logprobs'] if block['logprobs'] else None,
                    seed=seed
                )
                # Save the whole response object and the content separately
                trial_result['responses'].append({
                    'content': response.choices[0].message.content,
                    'full_response': response  # This saves the entire response
                })

            experiment_results.append(trial_result)

            # Judge evaluation, if initialized
            if initialize_judge and len(trial_result['responses']) >= 2:  # Ensure there are at least two responses
                # Extracting text content from responses for the judge's evaluation
                output_1 = trial_result['responses'][0]['content']
                output_2 = trial_result['responses'][1]['content']
                judge_message_content = f"You are a judge of two outputs, and you should evaluate both outputs based on {judge_criteria}. Here are the outputs: OUTPUT 1: {output_1} and OUTPUT 2: {output_2}. Please only output the single character 0,1,2,3,4, where 0 corresponds to OUTPUT 1 being significantly higher than OUTPUT 2 along the specified axis/axes ({judge_criteria}), 1 corresponds to somewhat higher, 2 to unsure/even, etc."
                
                judge_response = client.chat.completions.create(
                    model="gpt-4-1106-preview",  # Judge is always GPT-4
                    messages=[{"role": "system", "content": judge_message_content}],
                    temperature=0.5,
                    seed=123  # Consider using a seed relevant to the trial for reproducibility
                )
                trial_result['judge'] = judge_response.choices[0].message.content

        # Dictionary to map judge's decision to descriptive text
        judge_decision_map = {
            "0": "Response 1 significantly higher than Response 2 given criteria",
            "1": "Response 1 somewhat higher than Response 2 given criteria",
            "2": "Responses are equal or unsure given criteria",
            "3": "Response 2 somewhat higher than Response 1 given criteria",
            "4": "Response 2 significantly higher than Response 1 given criteria",
        }

        # Display experiment results
        for trial_index, trial_result in enumerate(experiment_results):
            with st.expander(f"Trial {trial_index + 1} Summary", expanded=False):
                st.subheader('**Overall result:**')
                # Displaying brief summary of responses
                for index, response in enumerate(trial_result['responses']):
                    st.write(f"Response {index + 1}: {response['content']}")
                if 'judge' in trial_result:
                    judge_explanation = judge_decision_map.get(trial_result['judge'], "Unknown decision")
                    st.write(f"**Judge's Decision:** {judge_explanation}")
                
                # Inner expander for detailed trial results
                st.subheader('**Detailed outputs:**')
                st.write(f"Seed for this trial: {trial_result['seed']}")
                for index, response in enumerate(trial_result['responses']):
                    st.write(f"Full Response {index + 1}:")
                    # Assuming 'full_response' is directly displayable; adjust if it's not directly printable
                    st.write(response['full_response'])

        import plotly.express as px
        from collections import Counter
        import pandas as pd

        # Assuming experiment_results is already populated and judge_decision_map is defined

        # Step 1: Collect Judge's Decisions
        judge_decisions = [trial_result['judge'] for trial_result in experiment_results if 'judge' in trial_result]

        # Step 2: Count the Frequency of Each Decision
        decision_counts = Counter(judge_decisions)

        # Prepare data for plotting: Labels and counts
        labels = [judge_decision_map.get(decision, "Unknown decision") for decision in decision_counts.keys()]
        counts = list(decision_counts.values())

        # Create a DataFrame for plotting
        df_judge_decisions = pd.DataFrame({
            'Decision Description': labels,
            'Count': counts
        })

        # Create the bar graph
        fig = px.bar(df_judge_decisions, x='Decision Description', y='Count', title="Frequency of Judge's Decisions")
        fig.update_layout(xaxis_title="Judge's Decision Description", yaxis_title="Count", xaxis={'categoryorder':'total descending'})

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.subheader('Analyses:')

    # Save the trial seeds and results for future reference
    st.session_state['trial_seeds'] = trial_seeds
    st.session_state['experiment_results'] = experiment_results

    # Check if the user selected 'qualitative analysis of results'
    if 'qualitative analysis of results' in st.session_state.selected_analyses and st.session_state.experiment_results:
        with st.spinner("Performing qualitative analysis..."):
            experiment_summary = ""
            for trial_result in st.session_state['experiment_results']:
                experiment_summary += f"Trial Seed: {trial_result['seed']}\n"
                for index, response in enumerate(trial_result['responses']):
                    agent_settings = st.session_state.agent_blocks[index]
                    # Include agent's system message
                    agent_system_message = agent_settings['system_message']
                    experiment_summary += f"Agent {index + 1} System Message: \"{agent_system_message}\", Model: {agent_settings['model']}, Temperature: {agent_settings['temperature']}, Presence Penalty: {agent_settings['presence_penalty']}, Logprobs: {'Enabled' if agent_settings['logprobs'] else 'Disabled'}, Response: {response['content']}\n"
                
                if 'judge' in trial_result:
                    judge_criteria = st.session_state.get('judge_criteria', 'Criteria not specified')  # Placeholder, adjust as needed
                    judge_system_message = f"Judge's Criteria: \"{judge_criteria}\". Evaluate based on the above criteria."
                    experiment_summary += judge_system_message + "\n"
                    experiment_summary += f"Judge's Decision: {trial_result['judge']}\n"
                    experiment_summary += f"Dict to understand judge's outputs: {judge_decision_map}"
                experiment_summary += "\n"  # Separate trials for clarity

            
            # Step 2: Call GPT for analysis
            qualitative_analysis_response = client.chat.completions.create(
                model="gpt-4-1106-preview",  # Use an appropriate GPT model
                messages=[{"role": "system", "content": f"Your job is to provide a qualitative analysis of the an experiment involving two or more LLMs with one or more initializations/independent variable differences. Your job is to intelligently infer the right amount of analysis and explanation to give of the experimental results so that the scientist who ran the experiment has a strong first-pass interpretation and understanding of what happened and any hypotheses as to why. Here is the data: {experiment_summary} YOUR OUTPUTS (format cleanly in markdown):"}],
                temperature=0.5
            )
            
            # Step 3: Display the analysis
            qualitative_analysis = qualitative_analysis_response.choices[0].message.content
            with st.expander("**Qualitative Analysis of Results**"):
                st.write(qualitative_analysis)
    # Assumed to be within the 'if st.button('Run Experiment'):' block or appropriately conditional block
                
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    # Check if 'differentiability analyses' is selected
    if 'differentiability analyses' in st.session_state.selected_analyses and st.session_state.experiment_results:
        with st.spinner("Performing differentiability analysis..."):    
            # Prepare embeddings and labels
            embeddings = []
            labels = []
            for trial_result in st.session_state.experiment_results:
                for index, response in enumerate(trial_result['responses']):
                    # Fetch embeddings for each response
                    embedding = get_embeddings(response['content'])
                    embeddings.append(embedding)
                    labels.append(index)

            # Convert lists to suitable numpy arrays for scikit-learn
            X = np.array(embeddings)
            y = np.array(labels)

            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate chance level for comparison
            chance_level = 1 / len(st.session_state.agent_blocks)

            # Display the modified accuracy message
            st.write(f"**A Random Forest Classifier can differentiate embeddings from the agent blocks with {accuracy*100:.2f}% accuracy (chance level would be {chance_level*100:.2f}%).**")

    from sklearn.decomposition import PCA
    import plotly.express as px
    import numpy as np

    # Check if 'results visualization' is selected
    if 'results visualization' in st.session_state.selected_analyses and st.session_state.experiment_results:
        with st.spinner("Generating results visualization..."):
            # Recompute embeddings for all responses
            embeddings = []
            texts = []  # To hold the original text for mouseover
            labels = []  # To hold the labels for color-coding

            for trial_result in st.session_state.experiment_results:
                for index, response in enumerate(trial_result['responses']):
                    embedding = get_embeddings(response['content'])
                    embeddings.append(embedding)
                    texts.append(response['content'])
                    labels.append(f"Agent Block {index + 1}")

            # Convert embeddings to a NumPy array
            X = np.array(embeddings)

            # Apply PCA to reduce to 3 dimensions
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)

            # Convert to a DataFrame for easier plotting
            df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
            df_plot['Agent Block'] = labels  # Or a list of names corresponding to labels for clarity
            df_plot['Text'] = texts  # The original response texts for hover information

            # Create a Plotly 3D scatter plot
            fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                                color='Agent Block', hover_name='Text',
                                title='Dimensionality-reduced visualization of LLM outputs',
                                width=800, height=600)  # Adjust width and height as needed

            # Ensure hover text is only visible on mouseover and reduce clutter
            fig.update_traces(marker=dict(size=5),
                            hoverinfo='name')

            # Use Streamlit to display the plot
            st.plotly_chart(fig, use_container_width=True)
