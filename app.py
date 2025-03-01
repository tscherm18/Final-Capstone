import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# Set page title and icon
st.set_page_config(page_title="Student Grade Predictor", page_icon='📚')

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions"])

# Load dataset
df = pd.read_csv('data/cleaned_data.csv')

# Home Page
if page == "Home":
    st.title("📊 Student Grade Predictor")
    st.subheader("Welcome to the Student Grade Predictor app!")
    st.write("""
        This app provides an interactive platform to explore the data of a group of students ranging from study habits, absences, extracurriculars, and various other features. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
    """)
    st.image('https://static.vecteezy.com/system/resources/previews/002/173/392/non_2x/student-studying-at-home-free-vector.jpg')
    st.write("Use the sidebar to navigate between different sections")


# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        This dataset contains comprehensive information on 2,392 high school students, detailing their demographics, study habits, parental involvement, 
        extracurricular activities, and academic performance. The target variable, GradeClass, classifies students' grades into distinct categories, 
        providing us a dataset that can be used for educational research, predictive modeling, and statistical analysis.
        
    """)
    st.image('https://cdn.corporatefinanceinstitute.com/assets/10-Poor-Study-Habits-Opener.jpeg', caption = 'Everyone has had the feeling')

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots'])

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Grade"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='GradeClass', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            df['GradeClass'] = pd.Categorical(df['GradeClass'], categories = ['A', 'B', 'C', 'D', 'F'], ordered = True)
            df1 = df.sort_values('GradeClass')
            st.plotly_chart(px.box(df1, x='GradeClass', y=b_selected_col, title=chart_title))


    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            if st.checkbox("Show by GradeClass"):
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='GradeClass', title=chart_title))
            else:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title))


# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = ['GradeClass', 'Ethnicity', 'Gender', 'GPA', 'Age'])
    y = df['GradeClass']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=3, max_value=16, value=7,)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.3f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")
    st.write(f"Precision Score: {precision_score(y_test, y_pred, average = 'macro'):.4f}")
    st.write(f"Recall Score: {recall_score(y_test, y_pred, average = 'macro'):.4f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

# Make Predictions Page
elif page == "Make Predictions":
    st.title("📚 Make Predictions")

    st.subheader("Adjust the values below to make predictions on the quality of air in your area:")

    # User inputs for prediction
    parentaledu = st.slider("Parental Education", min_value=0, max_value=4, value=2)
    study = st.slider("Study Time Weekly (hrs)", min_value=0, max_value=20, value=10)
    absence = st.slider("Total Absences", min_value=0, max_value=30, value=14)
    tutor = st.slider("Tutoring", min_value=0, max_value=1, value=0)
    support= st.slider("Parental Support", min_value=0, max_value=4, value=2)
    extra = st.slider("Extracurriculars", min_value=0, max_value=1, value=0)
    sport = st.slider("Sports", min_value=0, max_value=1, value=0)
    music = st.slider("Music", min_value=0, max_value=1, value=0)
    volunteer = st.slider("Volunteering", min_value=0, max_value=1, value=0)
    neighbors = st.slider("Select amount of KNN neighbors", min_value = 1, max_value = 19, value = 5)

    # User input dataframe
    user_input = pd.DataFrame({
        'ParentalEducation': [parentaledu],
        'StudyTimeWeekly': [study],
        'Absences': [absence],
        'Tutoring': [tutor],
        'ParentalSupport': [support],
        'Extracurricular': [extra],
        'Sports': [sport],
        'Music': [music],
        'Volunteering': [volunteer]

    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Use KNN (k=9) as the model for predictions
    model = KNeighborsClassifier(n_neighbors=neighbors)
    features = ['ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
                'Extracurricular', 'Sports', 'Music', 'Volunteering']
    X = df[features]
    y = df['GradeClass']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]

    # Display the result
    st.write(f"The model predicts that this student will receive the following grade: **{prediction}**")
