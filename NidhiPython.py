import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read the dataset
file_path = r"C:\Users\Nidhi Rana\Downloads\Air_Quality (2).csv"
df = pd.read_csv(file_path)

# Display missing values
print("Missing values in each column:\n", df.isnull().sum())

# Fill missing values (example: fill with mean for numerical columns)
df_filled = df.fillna(df.mean(numeric_only=True))

#Shape of the dataset
print("\nShape of the dataset:", df.shape)

#Display first 5 rows
print("\nFirst 5 rows:\n", df.head())

#Display last 5 rows
print("\nLast 5 rows:\n", df.tail())

#Information about dataset
print("\nDataset Information:")
print(df.info())

#Statistical summary
print("\nStatistical Summary:\n", df.describe())

#Columns in the dataset
print("\nColumns in dataset:\n", df.columns.tolist())

#Data types of each column
print("\nData types of columns:\n", df.dtypes)

#Index information
print("\nIndex info:\n", df.index)

#Display a random sample of 5 rows
print("\nRandom sample rows:\n", df.sample(5))

#Drop rows with missing values
df_dropped = df.dropna()

#Calculate mean of numerical columns
print("\nMean of columns:\n", df.mean(numeric_only=True))

#Calculate mode of columns
print("\nMode of columns:\n", df.mode().iloc[0])

#Calculate median of columns
print("\nMedian of columns:\n", df.median(numeric_only=True))


#1 ⿡ Objective: Create a Histogram of 'Data Value'

plt.figure(figsize=(10,6))

# Plot histogram
n, bins, patches = plt.hist(
    df['Data Value'].dropna(),
    bins=30,
    edgecolor='black',
    color='navy',   
    alpha=0.8       
)

# Plot black curve on top
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.plot(bin_centers, n, color='black', linewidth=3) 

# Titles and labels
plt.title('Distribution of Data Value', fontsize=18, fontweight='bold')
plt.xlabel('Data Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️⃣ Objective: Create a Bar Chart of Top 10 Geo Places by Average Data Value

top_places = df.groupby('Geo Place Name')['Data Value'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_places.index, y=top_places.values, hue=top_places.index, dodge=False, palette='viridis', edgecolor='black', legend=False)
plt.title('Top 10 Geo Places by Average Data Value', fontsize=16, fontweight='bold')
plt.xlabel('Geo Place Name')
plt.ylabel('Average Data Value')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3 ️⃣ Objective: Create a Scatter plot for Data Value Vs Time Period


plt.figure(figsize=(12,6))

norm = plt.Normalize(df['Data Value'].min(), df['Data Value'].max())
colors = plt.cm.rainbow(norm(df['Data Value']))

plt.scatter(
    df['Time Period'],
    df['Data Value'],
    c=colors,
    edgecolor='black',
    s=70,
    alpha=0.8
)

plt.title('Scatter Plot: Data Value vs Time Period', fontsize=16, fontweight='bold')
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Data Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')
if len(df['Time Period']) > 20:
    step = len(df['Time Period']) // 20
    plt.xticks(df['Time Period'][::step])

plt.tight_layout()
plt.show()


# 4️⃣ Objective: Plot a box plot 


plt.style.use('dark_background')

np.random.seed(42)
data = [np.random.normal(loc, 1.0, 100) for loc in [2, 6, 12]]

fig, ax = plt.subplots()

bp = ax.boxplot(data, vert=False, patch_artist=True)

# Only three colors: purple, blue, grey
custom_colors = ['purple', 'blue', 'grey']

for patch in bp['boxes']:
    patch.set_facecolor('none')
    bbox = patch.get_path().get_extents()
    x0, y0 = bbox.xmin, bbox.ymin
    width = bbox.width
    height = bbox.height

    num_colors = len(custom_colors)
    for i, color in enumerate(custom_colors):
        ax.add_patch(plt.Rectangle(
            (x0, y0 + (i / num_colors) * height),
            width,
            height / num_colors,
            color=color,
            lw=0
        ))

for whisker in bp['whiskers']:
    whisker.set_color('white')
for cap in bp['caps']:
    cap.set_color('white')
for median in bp['medians']:
    median.set_color('white')
for flier in bp['fliers']:
    flier.set(markerfacecolor='white', marker='o', alpha=0.5)

ax.set_title('Outlier Analysis of Air Quality Data', color='white')

plt.show()


# 5️⃣ Objective: Perform a Z-Test 

from scipy import stats

sample_mean = df['Data Value'].mean()
population_mean = 50
sample_std = df['Data Value'].std()
n = df['Data Value'].count()

z_score = (sample_mean - population_mean) / (sample_std / (n**0.5))
p_value = stats.norm.sf(abs(z_score)) * 2

print("\nZ-Score:", z_score)
print("P-Value:", p_value)


# 6️⃣ Objective: Build Correlation Heatmaps to Study Relationships Among Pollutants


# Pivot the data
pivot_pollutants = df.pivot_table(
    values='Data Value',
    index=['Geo Place Name', 'Time Period'],
    columns='Measure'
)

# Fill missing values
pivot_filled = pivot_pollutants.fillna(0)

#Correlation matrix
corr_matrix = pivot_filled.corr()
fig, ax = plt.subplots(figsize=(14, 10))  # a bit wider

sns.heatmap(
    corr_matrix,
    cmap='RdYlBu',
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    linecolor='white',
    square=True,
    cbar_kws={'shrink': 0.8},
    annot_kws={"size": 7, "color": 'black'},  # smaller font inside
    ax=ax
)
plt.title('Correlation Heatmap of Air Quality Indicators',
          fontsize=18, fontweight='bold', pad=30)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.subplots_adjust(left=0.25, right=0.95, top=0.88, bottom=0.25)
plt.show()
