import matplotlib.pyplot as plt

# Index to name mapping
index_to_name = {
    1: "1",#청
    2: "2",#민
    3: "3",#선
    4: "4"#준
}

# Sample accuracies
#dict_items([('sun.csv', [82.72]), ('chung.csv', [78.84]), ('min.csv', [83.0]), ('jun.csv', [59.36])])
prediction_accuracies = [95.44,90.16,93.6,91.36]

# Calculating average accuracy
avg_accuracy = sum(prediction_accuracies) / len(prediction_accuracies)

# Index of the maximum and minimum accuracies
max_index = prediction_accuracies.index(max(prediction_accuracies))
min_index = prediction_accuracies.index(min(prediction_accuracies))

# Colors for bars
colors = ['#1876FB', 'orange', 'green', '#7161a0']

plt.figure(figsize=(10, 5))
labels = []

# Plotting bars and adding labels
for i in range(len(prediction_accuracies)):
    plt.bar(i + 1, prediction_accuracies[i], color=colors[i])
    labels.append(index_to_name[i + 1])
    plt.text(i + 1, prediction_accuracies[i] / 2, f'{prediction_accuracies[i]}%', 
             ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Adding "Best" text for maximum accuracy
    if i == max_index:
        plt.text(i + 1, prediction_accuracies[i] / 2 + 5, 'Best', 
                 ha='center', va='center', color='#be2e22', fontweight='bold', fontsize=16)
    
    # Adding "Worst" text for minimum accuracy
    if i == min_index:
        plt.text(i + 1, prediction_accuracies[i] / 2 + 5, 'Worst', 
                 ha='center', va='center', color='#020f37', fontweight='bold', fontsize=16)

# Setting y-axis range from 0 to 100
plt.ylim(0, 100)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')

# Adding text for average accuracy
plt.text(2.5, 10, f'Average Accuracy ({avg_accuracy}%)', 
         ha='center', va='center', fontsize=25, 
         bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

plt.title('Average Accuracy per Dataset')
plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
plt.savefig('average_accuracy_per_dataset_aug.jpg')
plt.show()
