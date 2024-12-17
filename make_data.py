path = '/kaggle/input/power-line-security-zone-vegetation-detection/train/train'
las_files = os.listdir(path)
for las_file in las_files:
    print(las_file)
    las_filepath = os.path.join(path, las_file)
    las = laspy.read(las_filepath)
    points = las.xyz - las.header.offset
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.voxel_down_sample(voxel_size=0.4)
    df = pd.DataFrame(np.asarray(downpcd.points), columns=['x', 'y', 'z'])
    df['file_name'] = las_file
    df.to_csv(f'{las_file}.csv', index=False)
        
    del downpcd
    del points
    del df
    gc.collect()

output_dir = '/kaggle/working'
files = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
files.remove('/kaggle/working/.virtual_documents')
n = len(files)
test_size = 0.1
test_part = max(1, int(test_size * n))
test_files, train_files = files[:test_part], files[test_part:]

def make_count(sub, files):
    count = 0
    counts = {}
    for file in files:
        df = pd.read_csv(file)
        n = len(df)
        counts[file.split('/')[-1]] = [count, count+n]
        count += n
    pd.DataFrame(counts).to_csv(f'{sub}_counts.csv', index=False)

make_count('train', train_files)
make_count('test', test_files)

from datasets import load_dataset

dataset = load_dataset("csv", data_files={"train": train_files, "test": test_files})
dataset.push_to_hub('power_line_lidar_data', token=)
