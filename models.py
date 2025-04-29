from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor
import matplotlib.dates as mdates


class PriceForecaster:
    def forecast(self, period_percentages, min_price, max_price):
       
        base_trend = np.linspace(min_price, max_price, len(period_percentages))
        noise = np.random.uniform(-0.01, 0.01, len(period_percentages)) * (max_price - min_price)
        return np.clip(base_trend + noise, min_price, max_price)


class ProjectSalesPredictor:
    def __init__(self):
        self.first_sale_model = None
        self.last_sale_model = None
        self.portion_model = None
        self.company_median_exp = 1
        self.known_companies = set()

        self.first_sale_features = [
            "days_until_agreement",
            "total_area",
            "duration",
            "company_exp",
            "log_total_area",
            "agreement_speed",
            "is_new_company",
        ]

        self.last_sale_features = self.first_sale_features.copy()

        self.portion_features = [
            "period_%",
            "total_area",
            "company_exp",
            "days_until_agreement",
            "duration",
            "month_sin",
            "month_cos",
            "project_month",
            "progress_rate",
            "is_new_company",
        ]

    def _prepare_data(self, df, is_training=True):
       
        df = df.copy()

        if is_training:
            self.known_companies = set(df["cons_company"].unique())
            self.company_median_exp = df.groupby("cons_company")["project_name"].nunique().median()

        df["is_new_company"] = 0
        if "cons_company" in df.columns:
            df["is_new_company"] = (~df["cons_company"].isin(self.known_companies)).astype(int)

        if "cons_company" in df.columns:
            df["company_exp"] = df.groupby("cons_company")["project_name"].transform("nunique")
            df["company_exp"] = df.apply(
                lambda x: self.company_median_exp if x["is_new_company"] else x["company_exp"],
                axis=1
            )
        else:
            df["company_exp"] = self.company_median_exp

        date_cols = ["start_date", "end_date", "agreement_date"]
        if "record_date" in df.columns:
            date_cols.append("record_date")
        df[date_cols] = df[date_cols].apply(pd.to_datetime, errors="coerce")

        
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df["project_duration_days"] = (df["end_date"] - df["start_date"]).dt.days

        if 'agreement_date' in df.columns and 'start_date' in df.columns:
            df["days_until_agreement"] = (df["agreement_date"] - df["start_date"]).dt.days
            df["days_until_agreement"] = df["days_until_agreement"].clip(lower=0)

        if "record_date" in df.columns and 'start_date' in df.columns and 'project_duration_days' in df.columns:
            df["period_%"] = ((df["record_date"] - df["start_date"]).dt.days / 
                             df["project_duration_days"]).clip(0, 1)

        if is_training and "period_%" in df.columns:
            targets = df.groupby("project_name").agg(
                first_sale_period=("period_%", lambda x: x[x > 0].min()),
                last_sale_period=("period_%", "max"),
                sales_months=("record_date", "count")
            ).reset_index()
            df = df.merge(targets, on="project_name")

            if "portion_sold" not in df.columns and "area" in df.columns and "total_area" in df.columns:
                df["portion_sold"] = df["area"] / df["total_area"]

        if "project_name" in df.columns:
            df["project_month"] = df.groupby("project_name").cumcount() + 1

        if "total_area" in df.columns:
            df["log_total_area"] = np.log1p(df["total_area"])

        if "duration" in df.columns and "days_until_agreement" in df.columns:
            df["agreement_speed"] = np.where(
                df["duration"] > 0,
                df["days_until_agreement"] / df["duration"],
                0
            )

        if "record_date" in df.columns:
            df["month"] = df["record_date"].dt.month
        else:
            df["month"] = 6
            
        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"]/12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"]/12)

        if "period_%" in df.columns and "project_month" in df.columns:
            df["progress_rate"] = np.where(
                df["project_month"] > 0,
                df["period_%"] / df["project_month"],
                0
            )
        else:
            df["progress_rate"] = 0.1

        return df

    def _train_model(self, X, y, groups, features):
   
        models = []
        scores = []
        
        group_kfold = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        
        for train_idx, test_idx in group_kfold.split(X, y, groups):
            X_train = X.iloc[train_idx][features]
            X_test = X.iloc[test_idx][features]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if "first_sale_period" in y.name:
                preds = np.clip(preds, 0, 1)
            elif "last_sale_period" in y.name:
                preds = np.clip(preds, 0, 1)
            
            scores.append(mean_absolute_error(y_test, preds))
            models.append(model)
        
        return models[np.argmin(scores)], np.mean(scores)

    def fit(self, df):
  
        if "record_date" not in df.columns:
            raise ValueError("Training data must contain 'record_date' column")
        if "area" not in df.columns and "portion_sold" not in df.columns:
            raise ValueError("Training data must contain either 'area' or 'portion_sold' column")
            
        df = self._prepare_data(df, is_training=True)
        
        required_targets = ["first_sale_period", "last_sale_period", "portion_sold"]
        missing_targets = [t for t in required_targets if t not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        
        df = df[df["first_sale_period"] >= 0]
        df = df[df["last_sale_period"] >= 0]
        df = df[df["first_sale_period"] <= df["last_sale_period"]]
        
        groups = df["cons_company"] if "cons_company" in df.columns else np.zeros(len(df))
        
        print(f"Training on {len(df['project_name'].unique())} projects with {len(df)} records")
        
        print("\nTraining first sale period model...")
        self.first_sale_model, first_mae = self._train_model(
            df, df["first_sale_period"], groups, self.first_sale_features
        )
        print(f"First sale period MAE: {first_mae:.4f}")
        
        print("\nTraining last sale period model...")
        self.last_sale_model, last_mae = self._train_model(
            df, df["last_sale_period"], groups, self.last_sale_features
        )
        print(f"Last sale period MAE: {last_mae:.4f}")
        
        print("\nTraining portion sold model...")
        self.portion_model, portion_mae = self._train_model(
            df, df["portion_sold"], groups, self.portion_features
        )
        print(f"Portion sold MAE: {portion_mae:.4f}")
        
        return self

    def predict(self, new_data, min_price=None, max_price=None):

        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame([new_data])
        
        df = self._prepare_data(new_data, is_training=False)
        
        for feature in set(self.first_sale_features + self.last_sale_features + self.portion_features):
            if feature not in df.columns:
                if feature == "period_%":
                    df["period_%"] = 0.3
                elif feature == "project_month":
                    df["project_month"] = 1
        
        first_period = np.clip(
            self.first_sale_model.predict(df[self.first_sale_features])[0],
            0, 1
        )
        last_period = np.clip(
            self.last_sale_model.predict(df[self.last_sale_features])[0],
            first_period, 1
        )
        
        project_duration = df["duration"].iloc[0] if "duration" in df.columns else 12
        first_month = first_period * project_duration
        last_month = last_period * project_duration
        
        months = max(1, min(project_duration, int((last_month - first_month) + 0.5)))
        timeline_months = np.linspace(first_month, last_month, months)
        timeline_percent = timeline_months / project_duration
        
        
        if min_price is not None and max_price is not None:
            price_forecaster = PriceForecaster()
            price_forecast = price_forecaster.forecast(timeline_percent, min_price, max_price)
        else:
            price_forecast = [1] * len(timeline_percent)  
        
        monthly_predictions = []
        for i, (month_num, period, price) in enumerate(zip(timeline_months, timeline_percent, price_forecast), 1):
            month_data = df.copy()
            month_data["period_%"] = period
            month_data["project_month"] = i
            month_data["progress_rate"] = period / i if i > 0 else 0
            
            if "start_date" in month_data.columns and "project_duration_days" in month_data.columns:
                month_date = month_data["start_date"].iloc[0] + timedelta(
                    days=period * month_data["project_duration_days"].iloc[0]
                )
                month_data["month"] = month_date.month
                month_data["month_sin"] = np.sin(2 * np.pi * month_data["month"]/12)
                month_data["month_cos"] = np.cos(2 * np.pi * month_data["month"]/12)
            
            portion = np.clip(
                self.portion_model.predict(month_data[self.portion_features])[0],
                0, 1
            )
            
            monthly_predictions.append({
                "month_num": i,
                "period_%": period,
                "portion_sold": portion,
                "area_sold": portion * month_data["total_area"].iloc[0] if "total_area" in month_data.columns else 0,
                "price_per_sqm": price,
                "monthly_revenue": portion * month_data["total_area"].iloc[0] * price if "total_area" in month_data.columns else 0,
                "date": month_date if "start_date" in month_data.columns and "project_duration_days" in month_data.columns else None,
                "project_month": month_num
            })
        
        sales_duration_days = (last_period - first_period) * (df["project_duration_days"].iloc[0] if "project_duration_days" in df.columns else 365)
        
        return {
            "first_sale_period": first_period,
            "last_sale_period": last_period,
            "sales_duration_days": sales_duration_days,
            "monthly_predictions": pd.DataFrame(monthly_predictions),
            "total_area": df["total_area"].iloc[0] if "total_area" in df.columns else 0,
            "total_predicted_revenue": sum(p["monthly_revenue"] for p in monthly_predictions),
            "avg_monthly_sales": sum(p["area_sold"] for p in monthly_predictions) / len(monthly_predictions) if monthly_predictions else 0
        }
    
    def predict_ongoing_project(self, project_data, completed_months, actual_sales=None, period_percentages=None):
      
        if not isinstance(project_data, pd.DataFrame):
            project_df = pd.DataFrame([project_data])
        else:
            project_df = project_data.copy()
        
   
        actual_sales_list = []
        if actual_sales:
            if isinstance(actual_sales, str):
                actual_sales_list = [float(x.strip()) for x in actual_sales.split(",")]
            else:
                actual_sales_list = list(actual_sales)
        
        period_pct_list = []
        if period_percentages:
            if isinstance(period_percentages, str):
                period_pct_list = [float(x.strip()) for x in period_percentages.split(",")]
            else:
                period_pct_list = list(period_percentages)
        
   
        if not period_pct_list and 'duration' in project_data:
            period_pct_list = [(i+1)/project_data['duration'] for i in range(completed_months)]
        
  
        historical_data = []
        for i in range(completed_months):
            month_data = {
                "project_name": project_data["project_name"],
                "record_date": pd.to_datetime(project_data["start_date"]) + pd.DateOffset(months=i),
                "area": actual_sales_list[i] if i < len(actual_sales_list) else np.nan,
                "period_%": period_pct_list[i] if i < len(period_pct_list) else (i+1)/project_data["duration"]
            }
            historical_data.append(month_data)
        
 
        if historical_data:
            hist_df = pd.DataFrame(historical_data)
            project_df = project_df.merge(hist_df, on="project_name", how="left")
        

        df = self._prepare_data(project_df, is_training=False)
        

        if period_pct_list:
            first_period = min(period_pct_list)
        else:
            first_period = np.clip(
                self.first_sale_model.predict(df[self.first_sale_features].iloc[[0]])[0],
                0, 0.3  
            )
        
 
        last_period = np.clip(
            self.last_sale_model.predict(df[self.last_sale_features].iloc[[0]])[0],
            first_period, 1  
        )
        

        if period_pct_list and max(period_pct_list) > last_period:
            last_period = min(max(period_pct_list) + 0.2, 1.0)
        
 
        project_duration = df["duration"].iloc[0] if "duration" in df.columns else 12
        first_month = first_period * project_duration
        last_month = last_period * project_duration
        

        full_months = max(1, int((last_month - first_month) + 0.5))
        full_timeline = np.linspace(first_period, last_period, full_months)
        
        if 'min_price' in project_data and 'max_price' in project_data:
            full_price_sim = self.price_forecaster.forecast(
                full_timeline,
                project_data['min_price'],
                project_data['max_price']
            )
        else:
            full_price_sim = np.linspace(1, 1.2, full_months) 
        
     
        full_price_dates = [
            pd.to_datetime(project_data["start_date"]) + timedelta(
                days=p * (df["project_duration_days"].iloc[0] if "project_duration_days" in df.columns else 365)
            )
            for p in full_timeline
        ]
        

        if period_pct_list:
            last_actual_period = max(period_pct_list)
            start_period = min(last_actual_period + (1/project_duration), 1.0)
            remaining_period = last_period - start_period
            
            remaining_months = max(1, int(np.ceil(remaining_period * project_duration)))
            timeline_percent = np.linspace(start_period, last_period, remaining_months)
            timeline_months = [completed_months + i + 1 for i in range(remaining_months)]
        else:
            timeline_months = np.linspace(completed_months + 1, last_month, full_months - completed_months)
            timeline_percent = timeline_months / project_duration
        

        total_sold_so_far = sum(actual_sales_list) if actual_sales_list else 0
        remaining_area = project_data["total_area"] - total_sold_so_far if "total_area" in project_data else 0
        
   
        monthly_predictions = []
        for i, (month_num, period) in enumerate(zip(timeline_months, timeline_percent), 1):
            month_data = df.copy()
            month_data["period_%"] = period
            month_data["project_month"] = month_num
            month_data["progress_rate"] = period / month_num if month_num > 0 else 0
            
            if "start_date" in month_data.columns and "project_duration_days" in month_data.columns:
                month_date = month_data["start_date"].iloc[0] + timedelta(
                    days=period * month_data["project_duration_days"].iloc[0]
                )
                month_data["month"] = month_date.month
                month_data["month_sin"] = np.sin(2 * np.pi * month_data["month"]/12)
                month_data["month_cos"] = np.cos(2 * np.pi * month_data["month"]/12)
            
            portion = np.clip(
                self.portion_model.predict(month_data[self.portion_features])[0],
                0, 1
            )
            
 
            price_idx = min(i-1, len(full_price_sim)-1)
            price = full_price_sim[price_idx]
            
            monthly_predictions.append({
                "month_num": month_num,
                "period_%": period,
                "portion_sold": portion,
                "area_sold": portion * month_data["total_area"].iloc[0] if "total_area" in month_data.columns else 0,
                "price_per_sqm": price,
                "monthly_revenue": portion * month_data["total_area"].iloc[0] * price if "total_area" in month_data.columns else 0,
                "date": month_date if "start_date" in month_data.columns and "project_duration_days" in month_data.columns else None,
                "project_month": month_num
            })
        

        if len(monthly_predictions) > 0:
            predicted_remaining = sum(p["area_sold"] for p in monthly_predictions)
            if predicted_remaining > 0 and remaining_area > 0:
                adjustment_factor = remaining_area / predicted_remaining
                for p in monthly_predictions:
                    p["area_sold"] = p["area_sold"] * adjustment_factor
                    if "total_area" in project_data:
                        p["portion_sold"] = p["area_sold"] / project_data["total_area"]
        

        historical_sales = []
        if actual_sales_list:
            for i, sales in enumerate(actual_sales_list):
                historical_sales.append({
                    "month_num": i+1,
                    "period_%": period_pct_list[i] if i < len(period_pct_list) else (i+1)/project_data["duration"] if "duration" in project_data else 0,
                    "area_sold": sales,
                    "price_per_sqm": full_price_sim[i] if i < len(full_price_sim) else np.nan,
                    "type": "actual"
                })
        

        predicted_sales = []
        for p in monthly_predictions:
            predicted_sales.append({
                "month_num": p["month_num"],
                "period_%": p["period_%"],
                "area_sold": p["area_sold"],
                "price_per_sqm": p["price_per_sqm"],
                "type": "predicted"
            })
        

        all_sales = historical_sales + predicted_sales
        sales_df = pd.DataFrame(all_sales).sort_values("month_num")
        
  
        completion_percentage = total_sold_so_far / project_data["total_area"] if "total_area" in project_data and project_data["total_area"] > 0 else 0
        sales_duration_days = (last_period - first_period) * (df["project_duration_days"].iloc[0] if "project_duration_days" in df.columns else 365)
        
        return {
            "first_sale_period": first_period,
            "last_sale_period": last_period,
            "sales_duration_days": sales_duration_days,
            "monthly_predictions": pd.DataFrame(monthly_predictions),
            "historical_sales": historical_sales,
            "all_sales": sales_df,
            "total_area": project_data["total_area"] if "total_area" in project_data else 0,
            "total_sold_so_far": total_sold_so_far,
            "remaining_area": remaining_area,
            "completion_percentage": completion_percentage,
            "avg_monthly_sales": sum(p["area_sold"] for p in monthly_predictions) / len(monthly_predictions) if monthly_predictions else 0,
            "full_price_simulation": list(zip(full_price_dates, full_price_sim)),
            "compressed_prediction": period_pct_list and max(period_pct_list) > last_period
        }


    
    def evaluate_and_plot(self, test_data, num_projects=3, figsize=(15, 10)):
    
        test_data = self._prepare_data(test_data, is_training=False)
        test_projects = test_data["project_name"].unique()
        num_projects = min(num_projects, len(test_projects))
        
        if len(test_projects) > num_projects:
            test_projects = np.random.choice(test_projects, size=num_projects, replace=False)
        
        results = []
        fig, axes = plt.subplots(num_projects, 1, figsize=figsize)
        
        if num_projects == 1:
            axes = [axes]
        
        for i, (project, ax) in enumerate(zip(test_projects, axes), 1):
            project_data = test_data[test_data["project_name"] == project].sort_values("period_%")
            total_area = project_data["total_area"].iloc[0] if "total_area" in project_data.columns else 0
            
            actual_first = project_data["period_%"].min()
            actual_last = project_data["period_%"].max()
            
            pred_first = np.clip(
                self.first_sale_model.predict(project_data[self.first_sale_features].iloc[[0]])[0],
                0, 1
            )
            pred_last = np.clip(
                self.last_sale_model.predict(project_data[self.last_sale_features].iloc[[0]])[0],
                pred_first, 1
            )
            
            pred_portions = np.clip(
                self.portion_model.predict(project_data[self.portion_features]),
                0, 1
            )
            pred_areas = pred_portions * total_area if total_area > 0 else pred_portions * 1
            actual_areas = project_data["area"] if "area" in project_data.columns else np.zeros(len(project_data))
            
            mae = mean_absolute_error(actual_areas, pred_areas)
            r2 = r2_score(actual_areas, pred_areas)
            
            results.append({
                "project_name": project,
                "mae": mae,
                "actual_first_period": actual_first,
                "actual_last_period": actual_last,
                "pred_first_period": pred_first,
                "pred_last_period": pred_last,
                "first_period_error": pred_first - actual_first,
                "last_period_error": pred_last - actual_last,
                "total_actual": actual_areas.sum(),
                "total_predicted": pred_areas.sum(),
                "months": len(project_data)
            })
            
            ax.plot(project_data["period_%"], actual_areas, "b-", 
                   linewidth=2, label="Actual Sales", 
                   marker="o", markersize=5)
            ax.plot(project_data["period_%"], pred_areas, "r--", 
                   linewidth=2, label="Predicted Sales", 
                   marker="s", markersize=4)
            
            ax.axvline(x=actual_first, color="g", linestyle="-", 
                      linewidth=1, label="Actual Start")
            ax.axvline(x=actual_last, color="m", linestyle="-", 
                      linewidth=1, label="Actual End")
            ax.axvline(x=pred_first, color="g", linestyle=":", 
                      linewidth=1.5, label="Predicted Start")
            ax.axvline(x=pred_last, color="m", linestyle=":", 
                      linewidth=1.5, label="Predicted End")
            
            ax.set_title(
                f"{project}\n"
                f"MAE: {mae:.1f} m² | R²: {r2:.2f} | Area: {total_area:,.0f} m²\n"
                f"First Sale Error: {(pred_first - actual_first):.3f} | "
                f"Last Sale Error: {(pred_last - actual_last):.3f}",
                fontsize=10, pad=10
            )
            ax.set_xlabel("Project Completion (%)", fontsize=9)
            ax.set_ylabel("Area Sold (m²)", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            
            if i == 1:
                ax.legend(fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        
        return pd.DataFrame(results), fig

    def plot_sales_forecast(self, predictions, project_data, title_suffix=""):
    
        fig = plt.figure(figsize=(16, 8))
        ax = plt.gca()
        
  
        start_date = pd.to_datetime(project_data['start_date'])
        end_date = pd.to_datetime(project_data['end_date'])
        project_days = (end_date - start_date).days
        
        monthly_data = predictions.get('monthly_predictions', pd.DataFrame())
        
        if 'date' not in monthly_data.columns:
            monthly_data['date'] = monthly_data['period_%'].apply(
                lambda p: start_date + timedelta(days=p*project_days))
        
      
        ax.plot(monthly_data['date'], monthly_data['area_sold'], 'bo--',
                markersize=8, linewidth=2.5, label='Forecasted Sales')
        
      
        lower = monthly_data['area_sold'] * 0.9
        upper = monthly_data['area_sold'] * 1.1
        ax.fill_between(monthly_data['date'], lower, upper,
                       color='blue', alpha=0.2,
                       label='±10% Uncertainty Range')
        
    
        if 'first_sale_period' in predictions:
            first_date = start_date + timedelta(days=predictions['first_sale_period']*project_days)
            ax.axvline(first_date, color='limegreen', linestyle='-', 
                       linewidth=3, alpha=0.7, label='Predicted First Sale')
        
        if 'last_sale_period' in predictions:
            last_date = start_date + timedelta(days=predictions['last_sale_period']*project_days)
            ax.axvline(last_date, color='red', linestyle='-', 
                       linewidth=3, alpha=0.7, label='Predcited Last Sale')
        
       
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        ax.set_xlim(start_date, end_date)
        title = f"Sales Forecast for {project_data['project_name']} {title_suffix}".strip()
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Area Sold (m²)', fontsize=12)
        
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper left', fontsize=10, framealpha=1)
        
        plt.tight_layout()
        return fig
        
    def plot_ongoing_project_results(self, ongoing_predictions, project_data):
       
        fig = plt.figure(figsize=(16, 8))
        ax = plt.gca()
    
    
        start_date = pd.to_datetime(project_data['start_date'])
        end_date = pd.to_datetime(project_data['end_date'])
        project_days = (end_date - start_date).days
    
        monthly_data = ongoing_predictions.get('monthly_predictions', pd.DataFrame())
        historical_sales = ongoing_predictions.get('historical_sales', [])
    
    
        hist_dates = []
        hist_values = []
        model_preds = [] 
        
        if historical_sales:
           
            temp_df = pd.DataFrame([project_data])
            temp_df = self._prepare_data(temp_df, is_training=False)
            
            for s in historical_sales:
                period = s['period_%']
                hist_dates.append(start_date + timedelta(days=period * project_days))
                hist_values.append(s['area_sold'])
                
                
                month_data = temp_df.copy()
                month_data["period_%"] = period
                month_data["project_month"] = s['month_num']
                month_data["progress_rate"] = period / s['month_num'] if s['month_num'] > 0 else 0
                
                if "start_date" in month_data.columns and "project_duration_days" in month_data.columns:
                    month_date = month_data["start_date"].iloc[0] + timedelta(
                        days=period * month_data["project_duration_days"].iloc[0]
                    )
                    month_data["month"] = month_date.month
                    month_data["month_sin"] = np.sin(2 * np.pi * month_data["month"]/12)
                    month_data["month_cos"] = np.cos(2 * np.pi * month_data["month"]/12)
                
                portion = np.clip(
                    self.portion_model.predict(month_data[self.portion_features])[0],
                    0, 1
                )
                model_pred = portion * month_data["total_area"].iloc[0] if "total_area" in month_data.columns else 0
                model_preds.append(model_pred)
    
     
        if 'date' not in monthly_data.columns and 'period_%' in monthly_data.columns:
            monthly_data['date'] = monthly_data['period_%'].apply(
                lambda p: start_date + timedelta(days=p * project_days))
    
   
        if hist_dates:
            ax.plot(hist_dates, hist_values, 'go-',
                    markersize=8, linewidth=2.5,
                    label='Actual Sales')
            
        
            ax.plot(hist_dates, model_preds, 'ro--',
                    markersize=6, linewidth=2,
                    label='Model Prediction (Historical)')
    
  
        if not monthly_data.empty:
            ax.plot(monthly_data['date'], monthly_data['area_sold'], 'bo--',
                    markersize=8, linewidth=2.5,
                    label='Forecasted Sales')
    
   
            last_sale_period = ongoing_predictions.get('last_sale_period', 1.0)
            last_sale_date = start_date + timedelta(days=last_sale_period * project_days)
            ax.axvline(x=last_sale_date, color='orange', linestyle='--', linewidth=2,
                      label=f'Predicted Last Sale')
    
      
            if hist_dates:
                ax.plot([hist_dates[-1], monthly_data['date'].iloc[0]],
                        [hist_values[-1], monthly_data['area_sold'].iloc[0]],
                        'k--', linewidth=1.5, alpha=0.7, label='_nolegend_')
    
        
            lower = monthly_data['area_sold'] * 0.9
            upper = monthly_data['area_sold'] * 1.1
            ax.fill_between(monthly_data['date'], lower, upper,
                            color='blue', alpha=0.2,
                            label='±10% Forecast Range')
    

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
    
        ax.set_xlim(start_date, end_date)
        title = f"Sales Forecast for {project_data.get('project_name', 'Project')}"
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Area Sold (m²)', fontsize=12)
    
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper left', fontsize=10, framealpha=1)
    
        plt.tight_layout()
        return fig
    def evaluate_project(self, project_data):
       
        project_data = self._prepare_data(project_data, is_training=False)
        total_area = project_data["total_area"].iloc[0]
        

        pred_portions = np.clip(
            self.portion_model.predict(project_data[self.portion_features]),
            0, 1
        )
        pred_areas = pred_portions * total_area
        

        return {
            'mae': mean_absolute_error(project_data["area"], pred_areas),
            'first_period_error': (self.first_sale_model.predict(project_data[self.first_sale_features].iloc[[0]])[0] - 
                                  project_data["period_%"].min()),
            'last_period_error': (self.last_sale_model.predict(project_data[self.last_sale_features].iloc[[0]])[0] - 
                                 project_data["period_%"].max())
        }

class MonteCarloSimulator:
    def __init__(self, sales_predictor):
        self.sales_predictor = sales_predictor
        self.price_forecaster = PriceForecaster()
    
    def run_simulation(self, project_data, min_price, max_price, 
                      cost_official, unfin_cost, num_simulations=10000):
       
        predictions = self.sales_predictor.predict(project_data, min_price, max_price)
        forecast_df = predictions['monthly_predictions']
        duration = project_data['duration'] if 'duration' in project_data else 12
        
        forecast_df = forecast_df[forecast_df['project_month'] <= duration]
        
        price_forecast = forecast_df['price_per_sqm'].values if 'price_per_sqm' in forecast_df.columns else np.ones(len(forecast_df))
        sales_forecast = forecast_df['area_sold'].values if 'area_sold' in forecast_df.columns else np.zeros(len(forecast_df))
        
        sales_lower = sales_forecast * 0.9
        sales_upper = sales_forecast * 1.1
        price_lower = price_forecast * 0.9
        price_upper = price_forecast * 1.1
        
        Capex_mean = (cost_official - unfin_cost) / duration if duration > 0 else 0
        Capex_lower = Capex_mean * 0.9
        Capex_upper = Capex_mean * 1.1
        
        np.random.seed(42)
        total_cash_flows = []
        
        for _ in range(num_simulations):
            sales_coef = np.random.uniform(0.9, 1.1, size=len(sales_forecast))
            price_coef = np.random.uniform(0.9, 1.1, size=len(price_forecast))
            capex_coef = np.random.uniform(0.9, 1.1)
            
            sales_sim = sales_forecast * sales_coef
            price_sim = price_forecast * price_coef
            capex_sim = Capex_mean * capex_coef
            
            CFi = sales_sim * price_sim - capex_sim
            total_cash_flows.append(np.sum(CFi))
        
        total_cash_flows = np.array(total_cash_flows)
        
        failure_rate = np.sum(total_cash_flows <= 0) / num_simulations
        mean_cash_flow = np.mean(total_cash_flows)
        std_dev_cash_flow = np.std(total_cash_flows)
        
        self._plot_monte_carlo_results(
            total_cash_flows, mean_cash_flow, std_dev_cash_flow,
            failure_rate, project_data.get('project_name', 'Project'),
            cost_official, unfin_cost
        )
        
        return {
            'failure_rate': failure_rate,
            'mean_cash_flow': mean_cash_flow,
            'std_dev_cash_flow': std_dev_cash_flow,
            'min_cash_flow': np.min(total_cash_flows),
            'max_cash_flow': np.max(total_cash_flows),
            'median_months_to_breakeven': self._calculate_breakeven_months(
                sales_forecast, price_forecast, Capex_mean, duration
            ),
            'simulation_results': total_cash_flows
        }
    
    def _calculate_breakeven_months(self, sales, prices, monthly_cost, duration):
    
        cumulative_cash = np.cumsum(sales * prices - monthly_cost)
        breakeven_point = np.argmax(cumulative_cash >= 0)
        return min(breakeven_point + 1, duration) if breakeven_point > 0 else duration
    
    def _plot_monte_carlo_results(self, total_cash_flows, mean_cf, std_dev_cf, failure_rate,
                                project_name, cost_official, unfin_cost):
       
        fig = plt.figure(figsize=(12, 6)) 
        
      
        counts, bins, _ = plt.hist(total_cash_flows, bins=50, 
                                 density=True, alpha=0.6,
                                 color='green', edgecolor='black')
        
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_cf, std_dev_cf)
        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
        
       
        plt.axvline(mean_cf, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean ({mean_cf:,.0f})')
        plt.axvline(0, color='black', linestyle='-',
                   linewidth=1.5, label='Breakeven Point')
        
   
        failure_x = np.linspace(xmin, 0, 50)
        failure_p = stats.norm.pdf(failure_x, mean_cf, std_dev_cf)
        plt.fill_between(failure_x, failure_p, color='red', alpha=0.3,
                        label=f'Failure Risk ({failure_rate:.1%})')
        
       
        title = (f"Cash Flow Distribution - {project_name}\n"
                f"Mean: {mean_cf:,.0f} | Std Dev: {std_dev_cf:,.0f} | "
                f"Failure Risk: {failure_rate:.1%}")
        plt.title(title, fontsize=12, pad=20)
        
        plt.xlabel('Total Cash Flow ($)', fontsize=10)
        plt.ylabel('Probability Density', fontsize=10)
        
    
        plt.legend(fontsize=9, framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.4)
        
       
        info_text = [
            f"Total Investment: {cost_official:,.0f}",
            f"Unfinished Cost: {unfin_cost:,.0f}",
            f"Simulations: {len(total_cash_flows):,}"
        ]
        plt.annotate('\n'.join(info_text),
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
        
        plt.tight_layout()
        return fig
        
    def plot_price_diapason(self, forecast_df, project_name):
       
        fig = plt.figure(figsize=(12, 6))
             
        dates = forecast_df['date']
        prices = forecast_df['price_per_sqm']        
      
        lower_prices = prices * 0.9
        upper_prices = prices * 1.1
        
        plt.plot(dates, prices, 'b-', linewidth=2, 
                 label='Simulated Price Trend')
        plt.fill_between(dates, lower_prices, upper_prices, 
                       color='blue', alpha=0.2,
                       label='±10% Simulation Range')
        

        plt.title(f"Simulated Price Range for {project_name}", fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price per sqm ($)', fontsize=12)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        return fig
    def plot_ongoing_project_diapason(self, ongoing_predictions, project_data, min_price, max_price):
       
        fig = plt.figure(figsize=(12, 6))
        
    
        start_date = pd.to_datetime(project_data['start_date'])
        end_date = pd.to_datetime(project_data['end_date'])
        project_days = (end_date - start_date).days
        duration = project_data.get('duration', 12) 
        
        last_sale_period = ongoing_predictions.get('last_sale_period', 1.0)
        last_sale_date = start_date + timedelta(days=last_sale_period * project_days)
        
        months_to_last_sale = int((last_sale_date - start_date).days / (project_days / duration))
        months_to_last_sale = max(1, min(months_to_last_sale, duration))  
        

        sale_dates = [start_date + timedelta(days=(i/duration)*project_days) 
                     for i in range(1, months_to_last_sale+1)]
        
       
        base_prices = np.linspace(min_price, max_price, months_to_last_sale)
        
     
        lower_prices = base_prices * 0.9
        upper_prices = base_prices * 1.1

        plt.plot(sale_dates, base_prices, 'b-', linewidth=2,
                 label='Simulated Price Trend'.format(min_price, max_price))
        
        
   
        plt.fill_between(sale_dates, lower_prices, upper_prices,
                       color='blue', alpha=0.2,
                       label='±10% Simulation Range')
        

        plt.title(f"Динамика цен для {project_data['project_name']}", fontsize=14)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Цена за м²', fontsize=12)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
     
        plt.xlim(start_date, last_sale_date + timedelta(days=30))
        
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        return fig


if __name__ == "__main__":
    try:
        print("Loading data...")
        df = pd.read_excel('data2.xlsx')

        date_cols = ['start_date', 'end_date', 'agreement_date', 'record_date']
        df[date_cols] = df[date_cols].apply(pd.to_datetime)

        print("\nSplitting data into train and test sets...")
        projects = df['project_name'].unique()
        train_projects, test_projects = train_test_split(projects, test_size=0.2, random_state=42)
        train_df = df[df['project_name'].isin(train_projects)]
        test_df = df[df['project_name'].isin(test_projects)]

        print(f"\nTotal projects: {len(projects)}")
        print(f"Training projects: {len(train_projects)}")
        print(f"Test projects: {len(test_projects)}")

        print("\nTraining models...")
        predictor = ProjectSalesPredictor()
        predictor.fit(train_df)

        print("\n=== Model Performance Metrics ===")
        
 
        print("\n1. First Sale Period Prediction:")
        first_sale_metrics = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE': [
                train_metrics_df['first_period_error'].abs().mean(),
                test_metrics_df['first_period_error'].abs().mean()
            ],
            'MSE': [
                (train_metrics_df['first_period_error']**2).mean(),
                (test_metrics_df['first_period_error']**2).mean()
            ]
        })
        print(first_sale_metrics.to_string(index=False))
        

        print("\n2. Last Sale Period Prediction:")
        last_sale_metrics = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE': [
                train_metrics_df['last_period_error'].abs().mean(),
                test_metrics_df['last_period_error'].abs().mean()
            ],
            'MSE': [
                (train_metrics_df['last_period_error']**2).mean(),
                (test_metrics_df['last_period_error']**2).mean()
            ]
        })
        print(last_sale_metrics.to_string(index=False))
        

        print("\n3. Sales Volume Prediction:")
        sales_metrics = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE (m²)': [
                train_metrics_df['mae'].mean(),
                test_metrics_df['mae'].mean()
            ],
            'MAPE (%)': [
                (train_metrics_df['mae'] / train_df['total_area']).mean() * 100,
                (test_metrics_df['mae'] / test_df['total_area']).mean() * 100
            ],
            'RMSE (m²)': [
                np.sqrt((train_metrics_df['mae']**2).mean()),
                np.sqrt((test_metrics_df['mae']**2).mean())
            ]
        })
        print(sales_metrics.to_string(index=False, float_format="%.1f"))
        simulator = MonteCarloSimulator(predictor)

        print("\nExample 1: New Project Prediction")

        new_project = {
            "project_name": "Яссауи",
            "start_date": "2022-11-30",
            "end_date": "2024-12-30",
            "agreement_date": "2023-02-02",
            "total_area": 10795,
            "duration": 25,
            "cons_company": "ТОО \"Айзия KZ\""
        }
        print("\nMaking predictions with price forecast...")
        predictions = predictor.predict(new_project, min_price=400000, max_price=460000)

        print("\n=== Prediction Summary ===")
        print(f"Project: {new_project['project_name']}")
        print(f"Total Area: {predictions['total_area']:,.0f} m²")
        print(f"First Sale Period: {predictions['first_sale_period']:.2%}")
        print(f"Last Sale Period: {predictions['last_sale_period']:.2%}")

        print("\nGenerating sales forecast visualization...")
        sales_fig = predictor.plot_sales_forecast(predictions, new_project)
        plt.figure(sales_fig.number)
        plt.show()

        print("\nGenerating price simulation...")
        price_fig = simulator.plot_price_diapason(predictions['monthly_predictions'], new_project['project_name'])
        plt.figure(price_fig.number)
        plt.show()

        print("\nRunning Monte Carlo Simulation...")
        simulation_results = simulator.run_simulation(
            project_data=new_project,
            min_price=400000,
            max_price=460000,
            cost_official=6548335303,
            unfin_cost=389462312,
            num_simulations=10000
        )

        print("\nExample 2: Ongoing Project Prediction")
        ongoing_project = {
            "project_name": "4YOU Business - 1",
            "start_date": "2019-09-10",
            "end_date": "2022-01-10",
            "agreement_date": "2019-10-10",
            "total_area": 19824,
            "duration": 28,
            "cons_company": "ТОО Мереке Сервис НС"
        }
        completed_months = 8
        actual_sales = [1360.85007, 2517.27002, 1200.700005, 734.4100037,
                        998.2700043, 946.3999977, 274.9700012, 1344.429989]
        period_percentages = [0.037514654, 0.060961313, 0.103165299, 0.148886284,
                              0.168815944, 0.203985932, 0.297772567, 0.385697538]

        print("\nMaking predictions for ongoing project...")
        ongoing_predictions = predictor.predict_ongoing_project(
            project_data=ongoing_project,
            completed_months=completed_months,
            actual_sales=actual_sales,
            period_percentages=period_percentages
        )

        if 'price_per_sqm' not in ongoing_predictions['monthly_predictions']:
            ongoing_predictions['monthly_predictions']['price_per_sqm'] = np.linspace(
                501000, 580000, len(ongoing_predictions['monthly_predictions']))

        print("\nGenerating updated sales forecast visualization...")
        ongoing_sales_fig = predictor.plot_ongoing_project_results(ongoing_predictions, ongoing_project)
        plt.figure(ongoing_sales_fig.number)
        plt.show()

        print("\nGenerating ongoing price simulation...")
        ongoing_price_fig = simulator.plot_price_diapason(
            ongoing_predictions['monthly_predictions'],
            ongoing_project['project_name'],
            project_start_date=pd.to_datetime(ongoing_project['start_date']) 
        )
        plt.figure(ongoing_price_fig.number)
        plt.show()

        print("\nRunning Monte Carlo Simulation for ongoing project...")
        ongoing_simulation_results = simulator.run_simulation(
            project_data=ongoing_project,
            min_price=501000,
            max_price=580000,
            cost_official=12457003488,
            unfin_cost=47120000,
            num_simulations=10000
        )

    except FileNotFoundError:
        print("Error: Data file not found. Please ensure 'data2.xlsx' exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
