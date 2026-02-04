import flask
from flask import Flask, render_template

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'test'

@app.route('/')
def test_template():
    try:
        return render_template('dashboard.html',
            dataset_name="Test Dataset",
            total_records=100,
            stay_pct=80,
            leave_pct=20,
            models_compared=3,
            best_accuracy=85,
            overall_labels=["Stay", "Leave"],
            overall_values=[80, 20],
            dept_labels=["Sales", "R&D"],
            dept_values=[10, 10],
            job_labels=["Manager", "Dev"],
            job_values=[5, 15],
            tenure_labels=[1, 2, 3],
            tenure_values=[5.5, 10.2, 15.1],
            gender_labels=["Male", "Female"],
            gender_values=[12, 8],
            overtime_labels=["Yes", "No"],
            overtime_values=[10, 10],
            risk_bins_labels=["Low", "High"],
            risk_bins_values=[50, 50],
            risk_seg_labels=["Low", "Critical"],
            risk_seg_values=[40, 60],
            fi_labels=["Age", "Income"],
            fi_values=[0.5, 0.4],
            salary_leave_values=[{'x': 5000, 'y': 1}, {'x': 6000, 'y': 0}],
            salary_stay_values=[{'x': 8000, 'y': 0}, {'x': 9000, 'y': 1}],
            dept_health_labels=["Sales"],
            dept_health_values=[75],
            model_rows=[],
            model_names=["RF", "LR"],
            model_acc_values=[85, 80],
            model_f1_values=[0.8, 0.75],
            forecast_labels=["Jan", "Feb"],
            forecast_values=[10, 12],
            kpi_low_satisfaction=5,
            kpi_poor_wlb=10,
            kpi_overtime=15,
            kpi_no_promo=2,
            kpi_low_salary=8,
            stability_labels=["Q1", "Q2"],
            stability_values=[90, 85],
            age_bins=["20-30", "30-40"],
            age_counts=[20, 30],
            income_bins=["Low", "High"],
            income_counts=[40, 10],
            edu_labels=["BA", "MA"],
            edu_values=[30, 20],
            joblevel_labels=["L1", "L2"],
            joblevel_values=[25, 25],
            dept_names=["Sales"],
            dept_attrition=[5]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)

if __name__ == '__main__':
    with app.test_request_context():
        try:
            print(test_template())
        except Exception as e:
            pass
