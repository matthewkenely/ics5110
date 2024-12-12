import gradio as gr
from time import perf_counter
import pandas as pd

features = [
 'G1',
 'G2',
 'school',
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

def make_gradio(models):
    linear_regression_all = models[0]
    linear_regression_no_grades = models[1]
    linear_regression_only_grades = models[2]

    def predict(G1, G2, school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences):
        data = [G1, G2, school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences]
        data_df = pd.DataFrame([data], columns=features)

        all_df = data_df
        no_grades_df = data_df.drop(['G1', 'G2'], axis=1)
        only_grades_df = data_df[['G1', 'G2']]

        pred_all = linear_regression_all.predict(data_df)
        pred_no_grades = linear_regression_no_grades.predict(no_grades_df)
        pred_only_grades = linear_regression_only_grades.predict(only_grades_df)

        # Return Gradio dataframe. Rows are models, columns are predictions, runtime, and model size
        to_return = pd.DataFrame({
            'Model': ['All Features', 'No Grades', 'Only Grades'],
            'Prediction': [pred_all[0], pred_no_grades[0], pred_only_grades[0]],
        })

        return to_return

    # Convert to dictionary
    output = gr.Dataframe(headers=['Model', 'Predicted G3'], type='numpy')

    inputs = [
        gr.Slider(label="G1", minimum=0, maximum=20, step=1),
        gr.Slider(label="G2", minimum=0, maximum=20, step=1),
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
        title='Linear Regression',
        description='Enter the student features to predict the final grade',
        flagging_mode='never',
        live=True,
    )

    interface.launch(share=False, inline=True, debug=True)