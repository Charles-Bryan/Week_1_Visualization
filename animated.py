import numpy as np
import ffmpeg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

## Importing/Manipulating Data----------------------------------------------------------------------

tips = sns.load_dataset("tips")
tips = tips[["total_bill", "tip", "sex"]]
tips.sort_values(by=['total_bill'], inplace=True)
tips = tips.groupby(['total_bill', 'sex'], as_index=False).mean()
data = tips.pivot_table(values='tip', index='total_bill', columns='sex')

# Plot1: Bad attempt
plt.title('Initial Attempt')
plt.plot(data)
plt.savefig('Initial Attempt.png')
plt.close()

## Plot2: Interpolation
data_int = data.interpolate(method='index', axis=0, limit_area='inside')
data_int.dropna(inplace=True)
plt.title('Interpolation')
plt.plot(data_int)
plt.savefig('Interpolation.png')
plt.close()

## Ugly, let's SMOOTH it using simple moving averages
# Not sure exactly why I have to do this... some nonsense about categoricalIndex rmeinding me of R factors again.
data_smooth = pd.DataFrame(data={'total_bill': data_int.index.values,
      'Male': data_int['Male'],
      'Female': data_int['Female']})
data_smooth.set_index('total_bill', inplace=True)
data_smooth['Male_rolling'] = data_smooth['Male'].rolling(10).mean()
data_smooth['Female_rolling'] = data_smooth['Female'].rolling(10).mean()

# Plot3: Smooth
data_smooth[['Male_rolling', 'Female_rolling']].plot()
plt.title('Smoothing')
plt.savefig('Smoothing.png')
plt.close()
# Still not perfect because moving average isn't a great smoothing choice due to the sparsity on the high-end.
# Also doing rolling mean with averaged tip per meal price is a very bad idea... Focusing on the plotting aspects

## Plot4: Pretty(Seaborn)

sex_colors = [(0/255, 215/255, 255/255), (255/255, 86/255, 157/255)]
sns.set_palette(sex_colors)

my_style1 = {'axes.facecolor': "#bfbfbf",
              "axes.edgecolor": "black",
              'axes.grid': True,
              'axes.axisbelow': True,
              'axes.labelcolor': '.05',
              'figure.facecolor': 'white',
              "grid.color": "white",
              'grid.linestyle': '-',
              'text.color': '.05',
              'xtick.color': '.05',
              'ytick.color': '.05',
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'lines.solid_capstyle': 'round',
              'patch.edgecolor': 'w',
              'patch.force_edgecolor': True,
              'image.cmap': 'rocket',
              'font.family': ['sans-serif'],
              'font.sans-serif': ['Arial',
                                  'DejaVu Sans',
                                  'Liberation Sans',
                                  'Bitstream Vera Sans',
                                  'sans-serif'],
              'xtick.bottom': True,
              'xtick.top': False,
              'ytick.left': True,
              'ytick.right': False,
              'axes.spines.left': True,
              'axes.spines.bottom': True,
              'axes.spines.right': False,
              'axes.spines.top': False}
sns.set_style(my_style1)

fig1, ax1 = plt.subplots()

plt.xlim(0, 47)
plt.ylim(0, 5.55)
plt.xticks(np.arange(0, 46, 5))
plt.yticks(np.arange(0, 6, 0.5))
sns.lineplot(x=data_smooth.index, y=data_smooth['Male_rolling'], ax=ax1, label='Male')
sns.lineplot(x=data_smooth.index, y=data_smooth['Female_rolling'], ax=ax1, label='Female')
ax1.lines[1].set_linestyle("--")
ax1.lines[1].set_linewidth(2)
ax1.set_xlabel('Total Meal Cost ($)',fontsize=12)
ax1.set_ylabel('Average Tips ($)',fontsize=12)
ax1.set_title("Tips vs Meal Costs")
ax1.legend(fancybox=True, framealpha=1, facecolor="white", edgecolor="black", loc='upper left')

# Add gender image
path1 = 'male_symbol.png'
path2 = 'female_symbol.png'
# Adjust offsets based on zooming
ab1 = AnnotationBbox(OffsetImage(plt.imread(path1), zoom=0.017),
                     (data_smooth.index[-1]+1.5, data_smooth['Male_rolling'].iloc[-1]+0.1), frameon=False)
ab2 = AnnotationBbox(OffsetImage(plt.imread(path2), zoom=0.017),
                     (data_smooth.index[-1]+1, data_smooth['Female_rolling'].iloc[-1]), frameon=False)
ax1.add_artist(ab1)
ax1.add_artist(ab2)

plt.savefig('Sexy Seaborn.png')
plt.close()
## Very Nice, but is that the end of extravagence for line plots? VIDEO!
# Plot5: VIDEOS!
def update_line(num, data, line1, line2, ab1, ab2):
    x_data = data.index.to_numpy()[:num]

    y1_data = data['Male_rolling'].to_numpy()[:num]
    y2_data = data['Female_rolling'].to_numpy()[:num]

    line1.set_data(x_data, y1_data)
    line2.set_data(x_data, y2_data)

    if x_data.size:
        # Adjust offsets based on zooming
        ab1.xybox = (x_data[-1] + 1.5, y1_data[-1] + 0.1)
        ab2.xybox = [x_data[-1] + 1, y2_data[-1]]

    return line1, line2, ab1, ab2


data_ani = data_smooth[['Male_rolling', 'Female_rolling']].dropna()

fig2, ax2 = plt.subplots()

plt.title('Tips vs Meal Cost')

# Get line objects
line1, = plt.plot([], [], color=sns.color_palette()[0], label='Male')
line2, = plt.plot([], [], color=sns.color_palette()[1], label='Female')

imagebox1 = OffsetImage(plt.imread(path1), zoom=0.017)
imagebox2 = OffsetImage(plt.imread(path2), zoom=0.017)
ab1 = AnnotationBbox(imagebox1, (0, 0), pad=0, frameon=False)
ab2 = AnnotationBbox(imagebox2, (0, 0), pad=0, frameon=False)
ax2.add_artist(ab1)
ax2.add_artist(ab2)

# Prep the plot
plt.xlim(0, 47)
plt.ylim(0, 5.55)
plt.xticks(np.arange(0, 46, 5))
plt.yticks(np.arange(0, 6, 0.5))
ax2.lines[1].set_linestyle("--")
ax2.lines[1].set_linewidth(2)
ax2.set_xlabel('Total Meal Cost ($)', fontsize=12)
ax2.set_ylabel('Average Tips ($)', fontsize=12)
ax2.set_title("Tips vs Meal Costs")
ax2.legend(fancybox=True, framealpha=1, facecolor="white", edgecolor="black", loc='upper left')

line_ani = animation.FuncAnimation(fig2,
                                   update_line,      # Function to call each frame. USed 'frames' iter as arg 1.
                                   data_ani.shape[0]+1,    # Number of Frames (an interable or generator could be used)
                                   fargs=(data_ani, line1, line2, ab1, ab2),  # additional arguments to update line function
                                   interval=0.1,      # Time delay(ms) between frames
                                   blit=True,
                                   repeat_delay=10000)        # Intention is to optimize drwing but may cause undesired layering

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=18000)
line_ani.save('Growing_Lines.mp4', writer=writer)

plt.savefig('Video Seaborn.png')
plt.close()
