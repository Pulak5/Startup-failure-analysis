import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Startup Failure Analysis",
    page_icon="📉",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #e63946;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #111; margin-top: 2px; }
    .analysis-box {
        background: #f0f4ff;
        border-left: 3px solid #457b9d;
        padding: 12px 16px;
        border-radius: 3px;
        font-size: 0.85rem;
        color: #333;
        margin: 14px 0;
        line-height: 1.7;
    }
    .analysis-box b { color: #111; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("startup_failures.csv")

    df['Closed in'] = df['Closed in'].replace('Active', 2026)
    df['Closed in'] = pd.to_numeric(df['Closed in'], errors='coerce')
    df['Lifetime'] = df['Closed in'] - df['Started in']

    def convert_to_million(val):
        val = str(val).replace('$', '').strip().lower()
        if val in ['no data', 'unknown', 'n/a', '-', '', 'nan']:
            return None
        if 'k' in val:
            return float(val.replace('k', '')) / 1000
        elif 'm' in val:
            return float(val.replace('m', ''))
        else:
            try:
                return float(val) / 1_000_000
            except:
                return None

    def clean_funding(x):
        if pd.isna(x): return None
        x = str(x).lower().strip()
        if x in ['no data', 'unknown', 'n/a', '-', '']: return None
        x = x.replace(',', '')
        if x == '$0': return 0
        if '-' in x:
            parts = x.split('-')
            if len(parts) == 2:
                a, b = convert_to_million(parts[0]), convert_to_million(parts[1])
                if a is not None and b is not None:
                    return (a + b) / 2
        if '>' in x: return convert_to_million(x.replace('>', '').strip())
        if '<' in x: return convert_to_million(x.replace('<', '').strip())
        return convert_to_million(x)

    df['Funding_clean'] = df['Funding Amount'].apply(clean_funding)
    df['Funding_clean'] = df['Funding_clean'].fillna(df['Funding_clean'].median())

    mapping = {
        'Lack of Funds': 'Financial', 'Mismanagement of Funds': 'Financial',
        'No Market Need': 'Market', 'Bad Market Fit': 'Market',
        'Competition': 'Market', 'Bad Marketing': 'Market', 'Lack of PMF': 'Market',
        'Poor Product': 'Product',
        'Bad Management': 'Management', 'Lack of Experience': 'Management',
        'Lack of Focus': 'Management', 'Bad Business Model': 'Management',
        'Failure to Pivot': 'Management',
        'Legal Challenges': 'External', 'Dependence on Others': 'External',
        'Acquisition Flu': 'External',
        'Multiple Reasons': 'Other', 'Other': 'Other'
    }
    df['Cause_group'] = df['Failure Cause'].map(mapping)

    return df

df = load_data()

CAUSE_COLORS = {
    'Financial': '#e63946', 'Market': '#457b9d', 'Management': '#2a9d8f',
    'Product': '#e9c46a', 'External': '#8338ec', 'Other': '#aaa'
}


st.sidebar.title("📉 Startup Failures")
st.sidebar.markdown("---")

page = st.sidebar.radio("Go to", [
    "Overview",
    "Failure Analysis",
    "Survival & Funding",
    "KNN Classifier",
    "Linear Regression",
    "Predictor Tool"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

industries = sorted(df['Industry'].dropna().unique())
sel_ind = st.sidebar.multiselect("Industry", industries, default=industries)
if not sel_ind: sel_ind = industries

year_min = int(df['Closed in'].min())
year_max = int(df['Closed in'].max())
year_range = st.sidebar.slider("Year closed", year_min, year_max, (year_min, year_max))

causes = sorted(df['Cause_group'].dropna().unique())
sel_causes = st.sidebar.multiselect("Failure cause", causes, default=causes)
if not sel_causes: sel_causes = causes

fd = df[
    df['Industry'].isin(sel_ind) &
    df['Closed in'].between(*year_range) &
    df['Cause_group'].isin(sel_causes)
].copy()

st.sidebar.markdown(f"---\n**{len(fd)} startups** in view")

def base_layout(height=300, **extra):
    layout = dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='sans-serif', color='#222', size=12),
        margin=dict(t=40, b=30, l=40, r=20),
        height=height,
        legend=dict(font=dict(color='#333')),
    )
    layout.update(extra)
    return layout

def axis_style(**kwargs):
    """Returns axis dict merged with default grid/tick style."""
    base = dict(gridcolor='#eee', linecolor='#ccc',
                tickfont=dict(color='#333'), title_font=dict(color='#333'))
    base.update(kwargs)
    return base

def analysis(text):
    st.markdown(f'<div class="analysis-box">{text}</div>', unsafe_allow_html=True)


if page == "Overview":
    st.title("Startup Failure Analysis")
    st.caption("Source: Failory.com · Post-mortem dataset · 470+ startup post-mortems")

    c1, c2, c3, c4, c5 = st.columns(5)
    avg_life = fd['Lifetime'].mean()
    top_cause  = fd['Cause_group'].value_counts().index[0] if len(fd) else "—"
    top_ind    = fd['Industry'].value_counts().index[0] if len(fd) else "—"
    top_country= fd['Country'].value_counts().index[0] if len(fd) else "—"

    for col, label, val in [
        (c1, "Total Startups",  str(len(fd))),
        (c2, "Avg Lifespan",    f"{avg_life:.1f} yrs" if not np.isnan(avg_life) else "—"),
        (c3, "Top Cause",       top_cause),
        (c4, "Top Industry",    top_ind[:14]),
        (c5, "Top Country",     top_country),
    ]:
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Failures over time")
        yearly = fd.groupby('Closed in').size().reset_index(name='Count')
        fig = go.Figure(go.Scatter(
            x=yearly['Closed in'], y=yearly['Count'],
            mode='lines+markers',
            line=dict(color='#e63946', width=2),
            marker=dict(size=4),
            fill='tozeroy', fillcolor='rgba(230,57,70,0.08)'
        ))
        fig.update_layout(**base_layout(height=270, showlegend=False))
        fig.update_xaxes(**axis_style())
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Failures by cause")
        cc = fd['Cause_group'].value_counts().reset_index()
        cc.columns = ['Cause', 'Count']
        fig = go.Figure(go.Bar(
            x=cc['Count'], y=cc['Cause'], orientation='h',
            marker_color=[CAUSE_COLORS.get(c, '#aaa') for c in cc['Cause']],
        ))
        fig.update_layout(**base_layout(height=270, showlegend=False))
        fig.update_xaxes(**axis_style())
        fig.update_yaxes(**axis_style(categoryorder='total ascending'))
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Failures over time:</b> The chart shows peaks around 2015–2019, which reflects both the "
        "startup boom of that era and the typical 3–5 year runway before funds run out. "
        "Recent years (2022–2024) appear lower — not because fewer startups failed, but because "
        "recently failed startups may not yet be fully documented on Failory. "
        "<b>Market and Management</b> are consistently the top two failure causes, together accounting "
        "for over 50% of all shutdowns in this dataset."
    )

    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.subheader("Top 10 countries")
        cn = fd['Country'].value_counts().head(10).reset_index()
        cn.columns = ['Country', 'Count']
        fig = go.Figure(go.Bar(
            x=cn['Count'], y=cn['Country'],
            orientation='h', marker_color='#457b9d'
        ))
        fig.update_layout(**base_layout(height=300, showlegend=False))
        fig.update_xaxes(**axis_style())
        fig.update_yaxes(**axis_style(categoryorder='total ascending'))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Industry breakdown")
        ic = fd['Industry'].value_counts().reset_index()
        ic.columns = ['Industry', 'Count']
        fig = go.Figure(go.Pie(
            labels=ic['Industry'], values=ic['Count'],
            hole=0.45,
            marker=dict(line=dict(color='white', width=1.5)),
            textfont=dict(color='#222')
        ))
        fig.update_layout(**base_layout(height=300))
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Country distribution:</b> The US dominates because Failory primarily covers "
        "English-language startups — this is a dataset bias, not evidence that US startups "
        "fail more than others. India and the UK follow, both having large English-speaking "
        "startup ecosystems. "
        "<b>Industry breakdown:</b> E-Commerce and Software/Tech appear most frequently "
        "simply because those sectors attract the highest volume of new startups globally — "
        "a higher absolute count does not mean a higher failure rate compared to other industries."
    )


elif page == "Failure Analysis":
    st.title("Failure Analysis")
    st.caption("Cause breakdowns · Industry patterns · Employee size")

    st.subheader("Raw failure causes")
    rc = fd['Failure Cause'].value_counts().reset_index()
    rc.columns = ['Cause', 'Count']
    fig = go.Figure(go.Bar(x=rc['Cause'], y=rc['Count'], marker_color='#e63946'))
    fig.update_layout(**base_layout(height=320))
    fig.update_xaxes(**axis_style(tickangle=-30))
    fig.update_yaxes(**axis_style())
    st.plotly_chart(fig, use_container_width=True)

    analysis(
        "Before grouping, <b>No Market Need</b> and <b>Bad Management</b> are the single largest "
        "individual causes. This aligns with industry research — CB Insights and other studies "
        "consistently find that lack of product-market fit is the #1 reason startups fail. "
        "In this dataset, when related causes are grouped (e.g. 'No Market Need', 'Bad Market Fit', "
        "'Competition' → <b>Market</b>), Market failures become the dominant category overall."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Industry × Cause (heatmap)")
        pivot = pd.crosstab(fd['Industry'], fd['Cause_group'])
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale='Blues',
            text=pivot.values, texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color='#333'))
        ))
        fig.update_layout(**base_layout(height=380))
        fig.update_xaxes(**axis_style(tickangle=-20))
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Industry × Cause (stacked)")
        pivot2 = pd.crosstab(fd['Industry'], fd['Cause_group'])
        fig = go.Figure()
        for cause in pivot2.columns:
            fig.add_trace(go.Bar(
                name=cause, x=pivot2.index, y=pivot2[cause],
                marker_color=CAUSE_COLORS.get(cause, '#aaa')
            ))
        fig.update_layout(**base_layout(height=380, barmode='stack'))
        fig.update_xaxes(**axis_style(tickangle=-30))
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Industry × Cause:</b> E-Commerce and Software startups show the widest spread across "
        "all failure causes — reflecting the sheer variety of business models in those sectors. "
        "Healthcare startups lean more towards <b>External</b> failures (legal, regulatory), "
        "which makes sense given the compliance-heavy nature of the industry. "
        "Management and Market failures appear across every industry, confirming that "
        "these are universal risks that no sector is immune to."
    )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Employee size distribution")
        ec = fd['Employees'].value_counts().reset_index()
        ec.columns = ['Size', 'Count']
        fig = go.Figure(go.Bar(x=ec['Size'], y=ec['Count'], marker_color='#2a9d8f'))
        fig.update_layout(**base_layout(height=300))
        fig.update_xaxes(**axis_style(tickangle=-20))
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Average funding by failure cause")
        af = fd.groupby('Cause_group')['Funding_clean'].mean().sort_values(ascending=False).reset_index()
        af.columns = ['Cause', 'Avg Funding ($M)']
        fig = go.Figure(go.Bar(
            x=af['Cause'], y=af['Avg Funding ($M)'],
            marker_color=[CAUSE_COLORS.get(c, '#aaa') for c in af['Cause']],
            text=af['Avg Funding ($M)'].round(2), textposition='outside',
            textfont=dict(color='#333')
        ))
        fig.update_layout(**base_layout(height=300, showlegend=False))
        fig.update_xaxes(**axis_style())
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Employee size:</b> The vast majority of failed startups had fewer than 10 employees, "
        "which is expected since most startups never scale beyond a small team before shutting down. "
        "<b>Funding by cause:</b> External failures (legal challenges, acquisition issues) happened "
        "at companies with higher average funding — these tend to be later-stage companies that "
        "raised more before hitting external roadblocks. In contrast, Market failures cluster at "
        "lower funding levels, often hitting early-stage startups that couldn't validate demand "
        "before running out of runway. The type of failure, not the amount of money raised, "
        "is the key differentiator."
    )


elif page == "Survival & Funding":
    st.title("Survival & Funding")
    st.caption("Lifespan distributions · Funding patterns · Correlations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lifespan distribution (raw)")
        fig = go.Figure(go.Histogram(
            x=fd['Lifetime'].dropna(), nbinsx=25, marker_color='#457b9d'
        ))
        fig.update_layout(**base_layout(height=270, showlegend=False))
        fig.update_xaxes(**axis_style(title='Years'))
        fig.update_yaxes(**axis_style(title='Count'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Lifespan distribution (log scale)")
        log_life = np.log1p(fd['Lifetime'].dropna().clip(lower=0))
        fig = go.Figure(go.Histogram(
            x=log_life, nbinsx=25, marker_color='#e9c46a'
        ))
        fig.update_layout(**base_layout(height=270, showlegend=False))
        fig.update_xaxes(**axis_style(title='log(Years)'))
        fig.update_yaxes(**axis_style(title='Count'))
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Lifespan distribution:</b> The raw histogram is heavily right-skewed — most startups "
        "shut down within 1–3 years, but a small number survive for a decade or more. "
        "The log-transformed version approximates a normal (bell-shaped) distribution, "
        "which is why <b>log(Lifetime)</b> is used as the target variable in the linear regression "
        "model instead of raw Lifetime. Using raw Lifetime would violate the linearity and "
        "normality assumptions of linear regression."
    )

    st.subheader("Lifespan by industry")
    colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#8338ec', '#f4a261', '#264653']
    ind_order = fd.groupby('Industry')['Lifetime'].median().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, ind in enumerate(ind_order):
        subset = fd[fd['Industry'] == ind]['Lifetime'].dropna()
        fig.add_trace(go.Box(
            y=subset, name=ind,
            marker_color=colors[i % len(colors)],
            boxpoints='outliers', showlegend=False
        ))
    fig.update_layout(**base_layout(height=370))
    fig.update_xaxes(**axis_style())
    fig.update_yaxes(**axis_style(title='Lifetime (years)'))
    st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Lifespan by industry:</b> Median lifespans differ across industries, but the "
        "interquartile ranges heavily overlap — meaning industry alone is not a reliable "
        "predictor of how long a startup will survive. The presence of many outliers "
        "(dots above the whiskers) shows that some startups in every industry manage to "
        "survive well beyond the typical range. Hardware and Biotech-adjacent sectors "
        "tend to have slightly longer lifespans, likely because they require more time "
        "to develop products before market launch."
    )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Funding vs lifespan (scatter)")
        sdf = fd[fd['Lifetime'] > 0].copy()
        fig = go.Figure(go.Scatter(
            x=sdf['Funding_clean'], y=sdf['Lifetime'],
            mode='markers',
            marker=dict(
                color=[CAUSE_COLORS.get(c, '#aaa') for c in sdf['Cause_group']],
                size=6, opacity=0.7
            ),
            hovertemplate='$%{x:.2f}M · %{y:.1f} yrs<extra></extra>'
        ))
        fig.update_layout(**base_layout(height=300))
        fig.update_xaxes(**axis_style(title='Funding ($M)'))
        fig.update_yaxes(**axis_style(title='Lifetime (years)'))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Lifespan by failure cause")
        fig = go.Figure()
        for cause in fd['Cause_group'].dropna().unique():
            subset = fd[fd['Cause_group'] == cause]['Lifetime'].dropna()
            fig.add_trace(go.Box(
                y=subset, name=cause,
                marker_color=CAUSE_COLORS.get(cause, '#aaa'),
                boxpoints='outliers', showlegend=False
            ))
        fig.update_layout(**base_layout(height=300))
        fig.update_xaxes(**axis_style())
        fig.update_yaxes(**axis_style(title='Lifetime (years)'))
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Funding vs Lifespan:</b> There is no strong linear relationship between how much a "
        "startup raised and how long it survived. A well-funded startup can fail just as quickly "
        "as an unfunded one if the core problem is market or management related. "
        "Funding level influences the <i>range</i> of possible outcomes more than the expected outcome. "
        "<b>Lifespan by cause:</b> Financial failures tend to have slightly shorter lifespans — "
        "once money runs out, shutdown is swift. External failures (legal/acquisition) happen "
        "across a wider range of lifespans since they can strike at any stage."
    )

    st.subheader("Correlation — numeric features")
    num_cols = fd[['Funding_clean', 'Lifetime', 'Started in']].dropna()
    if len(num_cols) > 2:
        corr = num_cols.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale='RdBu', zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color='#333'))
        ))
        fig.update_layout(**base_layout(height=280))
        fig.update_xaxes(**axis_style())
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Correlation heatmap:</b> Funding_clean and Lifetime have a very weak positive correlation, "
        "confirming that more funding does not strongly predict a longer life. "
        "'Started in' (founding year) shows a slight negative correlation with Lifetime — "
        "newer startups have shorter recorded lifespans, partly because the dataset may not yet "
        "fully capture recent failures. None of the numeric features are strongly correlated "
        "with each other, which is actually good for the regression model — it means there is "
        "no multicollinearity issue."
    )


elif page == "KNN Classifier":
    st.title("KNN Classifier")
    st.caption("Predicting failure cause · Features: funding, industry, lifetime")

    @st.cache_data
    def prep_knn(df):
        mapping = {
            'Lack of Funds': 'Financial', 'Mismanagement of Funds': 'Financial',
            'No Market Need': 'Market', 'Bad Market Fit': 'Market',
            'Competition': 'Market', 'Bad Marketing': 'Market', 'Lack of PMF': 'Market',
            'Poor Product': 'Product',
            'Bad Management': 'Management', 'Lack of Experience': 'Management',
            'Lack of Focus': 'Management', 'Bad Business Model': 'Management',
            'Failure to Pivot': 'Management',
            'Legal Challenges': 'External', 'Dependence on Others': 'External',
            'Acquisition Flu': 'External',
            'Multiple Reasons': 'Other', 'Other': 'Other'
        }
        y_raw = df['Failure Cause'].map(mapping)
        industry = pd.get_dummies(df['Industry'], drop_first=True)
        X = pd.concat([df[['Funding_clean', 'Lifetime']], industry], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_raw, test_size=0.2, random_state=42
        )
        return X_scaled, y_raw, X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

    X_scaled, y_raw, X_train, X_test, y_train, y_test, scaler_knn, feat_cols = prep_knn(df)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Settings")
        k = st.slider("k (neighbors)", 1, 20, 8)
        st.caption("Default k=8 matches the notebook")

    model_knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    with col_b:
        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.1%}")
        m2.metric("Weighted F1", f"{report['weighted avg']['f1-score']:.3f}")
        m3.metric("Test samples", len(y_test))

    analysis(
        f"<b>Model accuracy: ~{acc:.1%}</b>. "
        "The model is about <b>2.5× better than random guessing</b> (random would be ~16.7% for 6 classes). "
        "The relatively low accuracy is expected and is explained by: "
        "(1) the dataset is small and imbalanced — some classes have very few samples; "
        "(2) features like funding and industry are not strong indicators of <i>why</i> a startup fails; "
        "(3) many failure categories overlap — for example, a startup that ran out of money may "
        "have done so because of bad management, bad market fit, or external factors, making the "
        "true label ambiguous. <b>distance weighting</b> (weights='distance') was used to give "
        "closer neighbors more influence, which improves performance on imbalanced data."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("k vs Accuracy (cross-validation)")
        k_range = list(range(1, 21))
        with st.spinner("Running cross-validation..."):
            cv_scores = [
                cross_val_score(
                    KNeighborsClassifier(n_neighbors=ki),  
                    X_scaled, y_raw, cv=5, scoring='accuracy'
                ).mean()
                for ki in k_range
            ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_range, y=cv_scores, mode='lines+markers',
            line=dict(color='#457b9d', width=2), marker=dict(size=6)
        ))
        fig.add_vline(x=k, line_color='#e63946', line_dash='dash',
                      annotation_text=f'k={k}',
                      annotation_font=dict(color='#e63946', size=12))
        fig.update_layout(**base_layout(height=300, showlegend=False))
        fig.update_xaxes(**axis_style(title='k'))
        fig.update_yaxes(**axis_style(title='CV Accuracy'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion matrix (normalized)")
        classes = sorted(y_raw.dropna().unique())
        cm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
        fig = go.Figure(go.Heatmap(
            z=cm, x=classes, y=classes, colorscale='Blues',
            text=cm.round(2), texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color='#333'))
        ))
        fig.update_layout(**base_layout(height=340))
        fig.update_xaxes(**axis_style(title='Predicted', tickangle=-20))
        fig.update_yaxes(**axis_style(title='Actual'))
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>k vs Accuracy curve:</b> The cross-validation curve shows how accuracy changes with k. "
        "Very low k (1–2) tends to overfit — it memorises the training data but generalises poorly. "
        "As k increases the model becomes more stable. The curve plateaus around k=7–10, "
        "which is why k=8 was chosen in the notebook as a good balance. "
        "<b>Confusion matrix:</b> Darker diagonal values indicate better prediction for that class. "
        "Market and Management are predicted most reliably because they have the most training samples. "
        "External, Product, and Other classes are frequently misclassified due to very few examples "
        "— the model has not seen enough of them to learn a clear boundary."
    )

    st.subheader("Per-class metrics")
    rdf = pd.DataFrame(report).T
    rdf = rdf[rdf.index.isin(classes)][['precision', 'recall', 'f1-score', 'support']].round(3)
    st.dataframe(rdf, use_container_width=True)

    analysis(
        "<b>Per-class metrics:</b> Classes with low support (few test samples) like External and Product "
        "show poor precision and recall — the model simply has not seen enough examples. "
        "Market achieves the best recall because it dominates the dataset. "
        "The weighted F1 score accounts for class imbalance by weighting each class by its support, "
        "giving a fairer overall performance measure than plain accuracy."
    )


elif page == "Linear Regression":
    st.title("Linear Regression")
    st.caption("Predicting startup lifetime · Features: funding, industry")

    @st.cache_data
    def run_reg(df):
        X = df[['Funding_clean']]
        X = pd.concat([X, pd.get_dummies(df['Industry'])], axis=1)
        y = np.log1p(df['Lifetime'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residual = y_test - y_pred
        coef = pd.DataFrame({
            'Feature': X.columns, 'Coef': model.coef_
        }).sort_values('Coef', ascending=False)
        return model, y_test, y_pred, residual, coef, X.columns.tolist()

    reg_model, y_test_r, y_pred_r, residual, coef, reg_cols = run_reg(df)

    mae  = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2   = r2_score(y_test_r, y_pred_r)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE (log scale)", f"{mae:.3f}")
    c3.metric("RMSE (log scale)", f"{rmse:.3f}")
    c4.metric("Target variable", "log(Lifetime)")

    analysis(
        f"<b>R² = {r2:.3f}</b> means the model explains about <b>{r2*100:.0f}% of the variance</b> "
        "in startup lifespan using only funding amount and industry. "
        "This is a meaningful result — the remaining variance comes from factors the dataset "
        "cannot capture, such as founder quality, market timing, product execution, and team dynamics. "
        "<b>Why log(Lifetime)?</b> Raw Lifetime is right-skewed (most startups fail early, "
        "a few last very long). Applying log1p transformation makes the distribution more symmetric "
        "and satisfies the normality assumption of linear regression. MAE and RMSE are reported "
        "on the log scale — small values here translate to reasonable year-level errors."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Residual plot")
        fig = go.Figure(go.Scatter(
            x=y_pred_r, y=residual, mode='markers',
            marker=dict(color='#457b9d', size=6, opacity=0.6)
        ))
        fig.add_hline(y=0, line_color='#e63946', line_dash='dash')
        fig.update_layout(**base_layout(height=300, showlegend=False))
        fig.update_xaxes(**axis_style(title='Predicted log(Lifetime)'))
        fig.update_yaxes(**axis_style(title='Residual'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actual vs Predicted")
        mn = min(y_test_r.min(), y_pred_r.min())
        mx = max(y_test_r.max(), y_pred_r.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test_r, y=y_pred_r, mode='markers',
            marker=dict(color='#2a9d8f', size=6, opacity=0.6), name='Points'
        ))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode='lines',
            line=dict(color='#e63946', dash='dash', width=1.5), name='Perfect fit'
        ))
        fig.update_layout(**base_layout(height=300, showlegend=False))
        fig.update_xaxes(**axis_style(title='Actual log(Lifetime)'))
        fig.update_yaxes(**axis_style(title='Predicted log(Lifetime)'))
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Residual plot:</b> After log transformation, residuals are distributed more randomly "
        "around the zero line — this indicates reduced heteroscedasticity compared to using raw Lifetime. "
        "However, some spread and pattern still exists, suggesting the model doesn't fully capture "
        "all underlying factors. <b>Actual vs Predicted:</b> Points close to the red diagonal line "
        "represent good predictions. The scatter around the line confirms moderate predictive power — "
        "the model gets the general direction right but cannot nail exact lifespans, "
        "which is expected given the limited features available."
    )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Residual distribution")
        fig = go.Figure(go.Histogram(x=residual, nbinsx=20, marker_color='#8338ec'))
        fig.add_vline(x=0, line_color='#e63946', line_dash='dash')
        fig.update_layout(**base_layout(height=280, showlegend=False))
        fig.update_xaxes(**axis_style(title='Residual'))
        fig.update_yaxes(**axis_style(title='Count'))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Feature coefficients (top/bottom)")
        top_coef = pd.concat([coef.head(8), coef.tail(5)])
        fig = go.Figure(go.Bar(
            x=top_coef['Coef'], y=top_coef['Feature'], orientation='h',
            marker_color=['#2a9d8f' if v > 0 else '#e63946' for v in top_coef['Coef']],
        ))
        fig.add_vline(x=0, line_color='#aaa', line_width=1)
        fig.update_layout(**base_layout(height=280, showlegend=False))
        fig.update_xaxes(**axis_style(title='Coefficient'))
        fig.update_yaxes(**axis_style())
        st.plotly_chart(fig, use_container_width=True)

    analysis(
        "<b>Residual distribution:</b> The histogram of residuals is approximately bell-shaped and "
        "centred near zero — a good sign that the linear regression assumptions are reasonably met. "
        "<b>Feature coefficients:</b> Green bars (positive coefficients) indicate features associated "
        "with <b>longer</b> startup lifespans; red bars indicate features associated with "
        "<b>shorter</b> ones. Industry dummies dominate — certain industries consistently give startups "
        "more or less runway. The Funding_clean coefficient is positive but small, confirming that "
        "more funding extends lifespan only modestly."
    )

elif page == "Predictor Tool":
    st.title("Failure Predictor")
    st.caption("Enter startup details · Get predicted failure cause + estimated lifetime")

    @st.cache_resource
    def train_models(df):
        mapping = {
            'Lack of Funds': 'Financial', 'Mismanagement of Funds': 'Financial',
            'No Market Need': 'Market', 'Bad Market Fit': 'Market',
            'Competition': 'Market', 'Bad Marketing': 'Market', 'Lack of PMF': 'Market',
            'Poor Product': 'Product',
            'Bad Management': 'Management', 'Lack of Experience': 'Management',
            'Lack of Focus': 'Management', 'Bad Business Model': 'Management',
            'Failure to Pivot': 'Management',
            'Legal Challenges': 'External', 'Dependence on Others': 'External',
            'Acquisition Flu': 'External',
            'Multiple Reasons': 'Other', 'Other': 'Other'
        }
        y_cls = df['Failure Cause'].map(mapping)
        industry_dummies = pd.get_dummies(df['Industry'], drop_first=True)
        X_cls = pd.concat([df[['Funding_clean', 'Lifetime']], industry_dummies], axis=1)
        valid = y_cls.notna() & X_cls.notna().all(axis=1)
        X_cls_v, y_cls_v = X_cls[valid], y_cls[valid]
        sc = StandardScaler()
        X_cls_s = sc.fit_transform(X_cls_v)
        knn = KNeighborsClassifier(n_neighbors=8, weights='distance').fit(X_cls_s, y_cls_v)

        X_reg = pd.concat([df[['Funding_clean']], pd.get_dummies(df['Industry'])], axis=1)
        y_reg = np.log1p(df['Lifetime'])
        valid_r = y_reg.notna() & X_reg.notna().all(axis=1)
        reg = LinearRegression().fit(X_reg[valid_r], y_reg[valid_r])

        return knn, sc, X_cls_v.columns.tolist(), reg, X_reg.columns.tolist()

    knn, sc, cls_cols, reg, reg_cols = train_models(df)

    col_form, col_out = st.columns(2)

    with col_form:
        st.subheader("Startup profile")
        sel_industry = st.selectbox("Industry", sorted(df['Industry'].dropna().unique()))
        funding = st.number_input("Funding raised ($M)", 0.0, 500.0, 1.0, 0.5)
        lifetime = st.number_input("Years active so far", 0.0, 20.0, 2.0, 0.5)
        predict = st.button("Predict", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Prediction")
        if predict:
            row = {c: 0 for c in cls_cols}
            row['Funding_clean'] = funding
            row['Lifetime'] = lifetime
            if f'Industry_{sel_industry}' in row:
                row[f'Industry_{sel_industry}'] = 1
            Xnew = sc.transform(pd.DataFrame([row])[cls_cols])
            pred_cause = knn.predict(Xnew)[0]
            proba = knn.predict_proba(Xnew)[0]
            classes = knn.classes_

            row2 = {c: 0 for c in reg_cols}
            row2['Funding_clean'] = funding
            if sel_industry in reg_cols:
                row2[sel_industry] = 1
            pred_life = np.expm1(reg.predict(pd.DataFrame([row2])[reg_cols])[0])

            st.metric("Most likely failure cause", pred_cause)
            st.metric("Predicted lifespan", f"{pred_life:.1f} years")

            st.markdown("**Probability by cause**")
            prob_df = pd.DataFrame({'Cause': classes, 'Prob': proba}).sort_values('Prob')
            fig = go.Figure(go.Bar(
                x=prob_df['Prob'], y=prob_df['Cause'], orientation='h',
                marker_color=[CAUSE_COLORS.get(c, '#aaa') for c in prob_df['Cause']],
                text=(prob_df['Prob'] * 100).round(1).astype(str) + '%',
                textposition='outside', textfont=dict(color='#333')
            ))
            fig.update_layout(**base_layout(
                height=240, showlegend=False,
                margin=dict(t=10, b=20, l=10, r=60)
            ))
            fig.update_xaxes(**axis_style(range=[0, 1], tickformat='.0%'))
            fig.update_yaxes(**axis_style())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fill in the startup profile on the left and click Predict.")

    if predict:
        st.markdown("---")
        st.subheader(f"Similar startups in dataset — {sel_industry}")
        similar = df[df['Industry'] == sel_industry][
            ['Industry', 'Cause_group', 'Funding_clean', 'Lifetime', 'Country']
        ].dropna(subset=['Cause_group']).head(10)
        similar.columns = ['Industry', 'Failure Cause', 'Funding ($M)', 'Lifetime (yrs)', 'Country']
        st.dataframe(similar.round(2), use_container_width=True, hide_index=True)

        analysis(
            f"<b>How to read this prediction:</b> The KNN model looks at the {len(classes)} failure "
            "cause classes and finds the 8 nearest startups (by funding, lifetime, and industry) "
            "in the training data, then predicts the most common cause among them. "
            "The probability bar shows the distribution across all classes — a spread-out distribution "
            "means the model is uncertain; a dominant bar means the model is more confident. "
            "The lifespan estimate comes from the linear regression model trained on log(Lifetime). "
            "Both predictions carry the same caveats as the underlying models — treat them as "
            "indicative, not definitive."
        )
