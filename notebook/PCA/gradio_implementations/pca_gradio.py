import gradio as gr
from time import perf_counter
import pandas as pd

features = ['school',
 'sex',
 'age',
 'address',
 'famsize',
 'Pstatus',
 'Medu',
 'Fedu',
 'Mjob',
 'Fjob',
 'reason',
 'guardian',
 'traveltime',
 'studytime',
 'failures',
 'schoolsup',
 'famsup',
 'paid',
 'activities',
 'nursery',
 'higher',
 'internet',
 'romantic',
 'famrel',
 'freetime',
 'goout',
 'Dalc',
 'Walc',
 'health',
 'absences']

def make_gradio(models, pcas, model_sizes):
    normal_model = models[0]
    pca_model_1 = models[1]
    pca_model_2 = models[2]

    pca_1 = pcas[0]
    pca_2 = pcas[1]

    normal_model_size_kb = model_sizes[0]
    pca_model_1_size_kb = model_sizes[1]
    pca_model_2_size_kb = model_sizes[2]

    def predict(school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences):
        data = [school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences]
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
        pred_normal, runtime_normal = time_prediction(normal_model, data_df)
        
        # PCA model 1 prediction
        pred_pca_1, runtime_pca_1 = time_prediction(pca_model_1, data_df, pca_1.transform)
        
        # PCA model 2 prediction
        pred_pca_2, runtime_pca_2 = time_prediction(pca_model_2, data_df, pca_2.transform)

        # Return Gradio dataframe. Rows are models, columns are predictions, runtime, and model size
        to_return = pd.DataFrame({
            'Model': ['Normal', 'PCA 95%', 'PCA 90%'],
            'Prediction': [pred_normal, pred_pca_1, pred_pca_2],
            'Runtime (ms)': [runtime_normal, runtime_pca_1, runtime_pca_2],
            'Model Size (KB)': [normal_model_size_kb, pca_model_1_size_kb, pca_model_2_size_kb]
        })

        return to_return

    # Convert to dictionary
    output = gr.Dataframe(headers=['Model', 'Predicted G3', 'Runtime (ms)', 'Model Size (KB)'], type='numpy')

    inputs = [
        gr.Dropdown(label="School", choices=["GP", "MS"], type="index"),  # GP: 0, MS: 1
        gr.Dropdown(label="Sex", choices=["F", "M"], type="index"),  # F: 0, M: 1
        gr.Slider(label="Age", minimum=15, maximum=22, step=1),
        gr.Dropdown(label="Address", choices=["U", "R"], type="index"),  # U: 1, R: 0
        gr.Dropdown(label="Family Size", choices=["LE3", "GT3"], type="index"),  # LE3: 1, GT3: 0
        gr.Dropdown(label="Parent Status", choices=["T", "A"], type="index"),  # T: 1, A: 0
        gr.Slider(label="Mother's Education", minimum=0, maximum=4, step=1),
        gr.Slider(label="Father's Education", minimum=0, maximum=4, step=1),
        gr.Dropdown(label="Mother's Job", choices=["at_home", "health", "services", "teacher", "other"], type="index"),
        gr.Dropdown(label="Father's Job", choices=["at_home", "health", "services", "teacher", "other"], type="index"),
        gr.Dropdown(label="Reason for School Choice", choices=["home", "reputation", "course", "other"], type="index"),
        gr.Dropdown(label="Guardian", choices=["mother", "father", "other"], type="index"),
        gr.Slider(label="Travel Time", minimum=1, maximum=4, step=1),
        gr.Slider(label="Study Time", minimum=1, maximum=4, step=1),
        gr.Slider(label="Failures", minimum=0, maximum=4, step=1),
        gr.Dropdown(label="School Support", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Family Support", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Paid Classes", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Extra-curricular Activities", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Nursery School", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Wants Higher Education", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Internet Access", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Dropdown(label="Romantic Relationship", choices=["no", "yes"], type="index"),  # no: 0, yes: 1
        gr.Slider(label="Family Relationship Quality", minimum=1, maximum=5, step=1),
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
        title='Principal Component Analysis',
        description='Enter the student features to predict the final grade',
        flagging_mode='never',
        live=True,
    )

    interface.launch(share=False, inline=True, debug=True)