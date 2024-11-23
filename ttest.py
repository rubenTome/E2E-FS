import pandas as pd
from scipy import stats

#facer o analisis para cada conxunto de datos, 
#mirar significancia, si siguen unha distribucion normal, e reflexalo na tabla

fl16_fmnist   = pd.read_csv("results/final_mean10/merged/float16_merged_fmnist.csv")
fl32_fmnist   = pd.read_csv("results/final_mean10/merged/float32_merged_fmnist.csv")
fl16_mnist    = pd.read_csv("results/final_mean10/merged/float16_merged_mnist.csv")
fl32_mnist    = pd.read_csv("results/final_mean10/merged/float32_merged_mnist.csv")
fl16_cifar10  = pd.read_csv("results/final_mean10/merged/float16_merged_cifar10.csv")
fl32_cifar10  = pd.read_csv("results/final_mean10/merged/float32_merged_cifar10.csv")
fl16_colon    = pd.read_csv("results/final_mean10/merged/float16_merged_colon.csv")
fl32_colon    = pd.read_csv("results/final_mean10/merged/float32_merged_colon.csv")
fl16_leukemia = pd.read_csv("results/final_mean10/merged/float16_merged_leukemia.csv")
fl32_leukemia = pd.read_csv("results/final_mean10/merged/float32_merged_leukemia.csv")
fl16_lung181  = pd.read_csv("results/final_mean10/merged/float16_merged_lung181.csv")
fl32_lung181  = pd.read_csv("results/final_mean10/merged/float32_merged_lung181.csv")
fl16_lymphoma = pd.read_csv("results/final_mean10/merged/float16_merged_lymphoma.csv")
fl32_lymphoma = pd.read_csv("results/final_mean10/merged/float32_merged_lymphoma.csv")

fl16_microarray = pd.concat([fl16_colon, fl16_leukemia, fl16_lung181, fl16_lymphoma])
fl32_microarray = pd.concat([fl32_colon, fl32_leukemia, fl32_lung181, fl32_lymphoma])

fl16_images = pd.concat([fl16_fmnist, fl16_mnist, fl16_cifar10])
fl32_images = pd.concat([fl32_fmnist, fl32_mnist, fl32_cifar10])

merged_fl16 = pd.concat([fl16_microarray, fl16_images])
merged_fl32 = pd.concat([fl32_microarray, fl32_images])

pairs = [
    [fl16_mnist, fl32_mnist, "MNIST"],
    [fl16_fmnist, fl32_fmnist, "FMNIST"],
    [fl16_cifar10, fl32_cifar10, "CIFAR10"],
    [fl16_colon, fl32_colon, "COLON"],
    [fl16_leukemia, fl32_leukemia, "LEUKEMIA"],
    [fl16_lung181, fl32_lung181, "LUNG181"],
    [fl16_lymphoma, fl32_lymphoma, "LYMHOMA"],
    [fl16_images, fl32_images, "IMAGENES"], 
    [fl16_microarray, fl32_microarray, "MICROARRAY"], 
    [merged_fl16, merged_fl32, "MERGED"]
]

for i in range(len(pairs)):
    print(pairs[i][2])

    df16 = pairs[i][0]
    df32 = pairs[i][1]

    print("info fl16:\n", df16[["duration", "emissions", "accuracy", "feature_mask"]].describe())
    print("info fl32:\n", df32[["duration", "emissions", "accuracy", "feature_mask"]].describe())

    print("\nshapiro fl16 duration:", stats.shapiro(df16["duration"]))
    print("shapiro fl32 duration:", stats.shapiro(df32["duration"]))
    print("shapiro fl16 emissions:", stats.shapiro(df16["emissions"]))
    print("shapiro fl32 emissions:", stats.shapiro(df32["emissions"]))
    print("shapiro fl16 accuracy:", stats.shapiro(df16["accuracy"]))
    print("shapiro fl32 accuracy:", stats.shapiro(df32["accuracy"]))

    print("\nttest duration:", stats.ttest_rel(df16["duration"], df32["duration"]))
    print("ttest emissions:", stats.ttest_rel(df16["emissions"], df32["emissions"]))
    print("ttest accuracy", stats.ttest_rel(df16["accuracy"], df32["accuracy"]))

    print("\nwilcoxon duration:", stats.wilcoxon(df16["duration"], df32["duration"]))
    print("wilcoxon emissions:", stats.wilcoxon(df16["emissions"], df32["emissions"]))
    print("wilcoxon accuracy:", stats.wilcoxon(df16["accuracy"], df32["accuracy"]))

    print("--------------------------------------------------\n\n")