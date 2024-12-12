import gradio as gr
from time import perf_counter
import pandas as pd

def make_gradio(kept_features, model):
    features = kept_features

    def predict(school, sex, age, address, Medu, Fedu, Mjob, reason, guardian, traveltime, studytime, failures, schoolsup, higher, internet, romantic, freetime, goout, Dalc, Walc, health, absences):
        data = [school, sex, age, address, Medu, Fedu, Mjob, reason, guardian, traveltime, studytime, failures, schoolsup, higher, internet, romantic, freetime, goout, Dalc, Walc, health, absences]
        data_df = pd.DataFrame([data], columns=features)
        
        # Run predictions multiple times for averaging
        def time_prediction(model, data, transform=None):
            runs = 100
            total_time = 0
            for _ in range(runs):
                start = perf_counter()
                if transform:
                    pred = model.predict(transform(data))
                else:
                    pred = model.predict(data)
                total_time += (perf_counter() - start)
            avg_runtime = (total_time / runs) * 1000  # Average runtime in ms
            return pred[0], avg_runtime

        # Normal prediction
        pred, runtime = time_prediction(model, data_df)

        # Return Gradio dataframe. Rows are models, columns are predictions, runtime, and model size
        to_return = pd.DataFrame({
            'Model': ['Stacking Ensemble'],
            'Prediction': [pred],
            'Runtime (ms)': [runtime],
        })

        return to_return

    # Convert to dictionary
    output = gr.Dataframe(headers=['Model', 'Predicted G3', 'Runtime (ms)'], type='numpy')

    inputs = [
        gr.Dropdown(label="School", choices=["GP", "MS"], type="index"),  # GP: 0, MS: 1
        gr.Dropdown(label="Sex", choices=["F", "M"], type="index"),  # F: 0, M: 1
        gr.Slider(label="Age", minimum=15, maximum=22, step=1),
        gr.Dropdown(label="Address", choices=["U", "R"], type="index"),  # U: 1, R: 0
        # gr.Dropdown(label="Family Size", choices=["LE3", "GT3"], type="index"),  # LE3: 1, GT3: 0
        # gr.Dropdown(label="Parent Status", choices=["T", "A"], type="index"),  # T: 1, A: 0
        gr.Slider(label="Mother's Education", minimum=0, maximum=4, step=1),
        gr.Slider(label="Father's Education", minimum=0, maximum=4, step=1),
        gr.Dropdown(label="Mother's Job", choices=["at_home", "health", "services", "teacher", "other"], type="index"),
        # gr.Dropdown(label="Father's Job", choices=["at_home", "health", "services", "teacher", "other"], type="index"),
        gr.Dropdown(label="Reason for School Choice", choices=["home", "reputation", "course", "other"], type="index"),
        gr.Dropdown(label="Guardian", choices=["mother", "father", "other"], type="index"),
        gr.Slider(label="Travel Time", minimum=1, maximum=4, step=1),
        gr.Slider(label="Study Time", minimum=1, maximum=4, step=1),
        gr.Slider(label="Failures", minimum=0, maximum=4, step=1),
        gr.Dropdown(label="School Support", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        # gr.Dropdown(label="Family Support", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        # gr.Dropdown(label="Paid Classes", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        # gr.Dropdown(label="Extra-curricular Activities", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        # gr.Dropdown(label="Nursery School", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Wants Higher Education", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Internet Access", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Romantic Relationship", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        # gr.Slider(label="Family Relationship Quality", minimum=1, maximum=5, step=1),
        gr.Slider(label="Free Time", minimum=1, maximum=5, step=1),
        gr.Slider(label="Going Out", minimum=1, maximum=5, step=1),
        gr.Slider(label="Workday Alcohol Consumption", minimum=1, maximum=5, step=1),
        gr.Slider(label="Weekend Alcohol Consumption", minimum=1, maximum=5, step=1),
        gr.Slider(label="Health Status", minimum=1, maximum=5, step=1),
        gr.Slider(label="Absences", minimum=0, maximum=93, step=1),
        ]

    interface = gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=output,
        title='Ensemble Model',
        description='Enter the student features to predict the final grade',
        flagging_mode='never',
        live=True,
    )

    interface.launch(share=False, inline=True, debug=True)