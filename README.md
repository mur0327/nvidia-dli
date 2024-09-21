# Disaster Risk Monitoring Using Satellite Imagery
### [위성 데이터 활용 재난 위험 모니터링](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-ES-01+V1)

AI 융합대학과 엔비디아가 함께하는 AI 빌드업 트레이닝<br>
<br>
김대중컨벤션센터 2024-09-21-토

---

* #### 학습 목표
  - 재해 위험 모니터링을 위한 컴퓨터 비전의 응용
  - 지구 관측 위성에서 수집한 데이터 조작
  - 대용량 영상 데이터를 효율적으로 처리하는 방법
  - 딥 러닝 모델 개발 과제
  - 엔드투엔드 머신 러닝 워크플로

---

JupyterLab에서 API 키를 발급받아 NVIDIA GPU Cloud를 이용

NGC CLI and Docker Registry

```python
apikey='<<<FIXME>>>'
```

```python
# DO NOT CHANGE THIS CELL
# write the config file
config_dict={'apikey': apikey, 'format_type': 'json', 'org': 'nvidia'}
with open('config', 'w') as f: 
    f.write(';WARNING - This is a machine generated file.  Do not edit manually.\n')
    f.write(';WARNING - To update local config settings, see "ngc config set -h"\n')
    f.write('\n[CURRENT]\n')
    for k, v in config_dict.items(): 
        f.write(k+'='+v+'\n')

# preview the config file
!cat config
```

```python
# DO NOT CHANGE THIS CELL
# move the config file to ~/.ngc
!mkdir -p ~/.ngc & mv config ~/.ngc/
```

```python
# DO NOT CHANGE THIS CELL
# login to NGC's docker registry
!docker login -u '$oauthtoken' -p $apikey nvcr.io
```

#### 01. 재난 위험 모니터링 시스템과 데이터 전처리
  1. 재난 위험 모니터링
      * 홍수 감지
      * 위성 이미지
![orbits](https://github.com/user-attachments/assets/a6171b85-5ed9-4c66-9894-b9f9e427e375)<br>
        
      * 컴퓨터 비전
![computer_vision_tasks](https://github.com/user-attachments/assets/3396a89b-86b4-4b60-8fb8-3172f63623a6)<br>

  2. 딥 러닝 모델 훈련 워크플로
![ml_workflow](https://github.com/user-attachments/assets/2fe26489-1b8b-4503-a95f-e0c4ef7d118c)<br>

  3. 데이터셋 소개
     * Sentinel-1 Data Public Access
![ESA](https://www.esa.int/)에서 제공하는 공용 데이터를 활용<br>
      
     * 홍수 감지 데이터 분석
![input_and_mask](https://github.com/user-attachments/assets/e835ef30-faa5-4286-938b-56083009d737)<br>

지역별 타일 분류
```python
# DO NOT CHANGE THIS CELL
# define function to get extent of an image from catalog
def get_extent(file_path): 
    """
    This function returns the extent as [left, right, bottom, top] for a given image. 
    """
    # read catalog for image
    with open(file_path) as f: 
        catalog_json=json.load(f)
    coordinates=catalog_json['geometry']['coordinates'][0]
    coordinates=np.array(coordinates)
    # get boundaries
    left=np.min(coordinates[:, 0])
    right=np.max(coordinates[:, 0])
    bottom=np.min(coordinates[:, 1])
    top=np.max(coordinates[:, 1])
    return left, right, bottom, top
```

```python
# DO NOT CHANGE THIS CELL
# define function to plot by region
def tiles_by_region(region_name, plot_type='images'): 
    # set catalog and images/masks path
    catalog_dir=os.path.join(os.getenv('LOCAL_DATA_DIR'), 'catalog', 'sen1floods11_hand_labeled_source')
    if plot_type=='images': 
        dir=os.path.join(image_dir, 'all_images')
        cmap='viridis'
    elif plot_type=='masks': 
        dir=os.path.join(mask_dir, 'all_masks')
        cmap='gray'
    else: 
        raise Exception('Bad Plot Type')

    # initiate figure boundaries, which will be modified based on the extent of the tiles
    x_min, x_max, y_min, y_max=181, -181, 91, -91
    fig=plt.figure(figsize=(15, 15))
    ax=plt.subplot(111)
    
    # iterate through each image/mask and plot
    file_list=os.listdir(dir)
    for each_file in file_list:
        # check if image/mask is related to region and a .png file
        if (each_file.split('.')[-1]=='png') & (each_file.split('_')[0]==region_name): 
            # get boundaries of the image
            extent=get_extent(f"{catalog_dir}/{each_file.split('.')[0]}/{each_file.split('.')[0]}.json")
            x_min, x_max=min(extent[0], x_min), max(extent[1], x_max)
            y_min, y_max=min(extent[2], y_min), max(extent[3], y_max)
            image=mpimg.imread(f'{dir}/{each_file}')
            plt.imshow(image, extent=extent, cmap=cmap)

    # set boundaries of the axis
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.show()
    
tiles_by_region(region_name='Spain', plot_type='images')
```

![tiles_by_region](https://github.com/user-attachments/assets/f6adc076-b888-456b-9042-264110fafd10)

DALI를 이용한 데이터 전처리

딥 러닝 모델이 정확한 예측을 내리려면 엄청난 양의 데이터가 필요하며, 모델의 크기와 복잡성이 커짐에 따라 이러한 필요성은 더욱 중요해진다.

Data Augmentation (데이터 증강)

딥 러닝 모델은 정확한 결과를 얻기 위해 방대한 양의 데이터로 학습해야 한다.
데이터 증강은 기하학적 변형, 색상 변환, 노이즈 추가 등과 같이 데이터에 무작위 교란을 도입하여 데이터 세트의 크기를 인위적으로 늘린다.
이러한 교란은 예측에 더 강하고, 과적합을 피하고, 더 나은 정확도를 제공하는 모델을 생성하는 데 도움이 된다.
DALI를 사용하여 모델 학습에 도입할 데이터 증강(예: 자르기, 크기 조정, 뒤집기)을 시연한다.

```python
# DO NOT CHANGE THIS CELL
import random

@pipeline_def
def augmentation_pipeline():
    # use fn.readers.file to read encoded images and labels from the hard drive
    image_pngs, _=fn.readers.file(file_root=image_dir)
    # use the fn.decoders.image operation to decode images from png to RGB
    images=fn.decoders.image(image_pngs, device='cpu')
    
    # the same augmentation needs to be performed on the associated masks
    mask_pngs, _=fn.readers.file(file_root=mask_dir)
    masks=fn.decoders.image(mask_pngs, device='cpu')
    
    image_size=512
    roi_size=image_size*.5
    roi_start_x=image_size*random.uniform(0, 0.5)
    roi_start_y=image_size*random.uniform(0, 0.5)
    
    # use fn.resize to investigate an roi, region of interest
    resized_images=fn.resize(images, size=[512, 512], roi_start=[roi_start_x, roi_start_y], roi_end=[roi_start_x+roi_size, roi_start_y+roi_size])
    resized_masks=fn.resize(masks, size=[512, 512], roi_start=[roi_start_x, roi_start_y], roi_end=[roi_start_x+roi_size, roi_start_y+roi_size])
    
    # use fn.resize to flip the image
    flipped_images=fn.resize(images, size=[-512, -512])
    flipped_masks=fn.resize(masks, size=[-512, -512])
    return images, resized_images, flipped_images, masks, resized_masks, flipped_masks
```

```python
# DO NOT CHANGE THIS CELL
pipe=augmentation_pipeline(batch_size=batch_size, num_threads=4, device_id=0)
pipe.build()
augmentation_pipe_output=pipe.run()
```

```python
# DO NOT CHANGE THIS CELL
# define a function display images
augmentation=['original', 'resized', 'flipped']
def show_augmented_images(pipe_output):
    image_batch, resized_image_batch, flipped_image_batch, mask_batch, resized_mask_batch, flipped_mask_batch=pipe_output
    columns=6
    rows=batch_size
    # create plot
    fig=plt.figure(figsize=(15, (15 // columns) * rows))
    gs=gridspec.GridSpec(rows, columns)
    grid_data=[image_batch, resized_image_batch, flipped_image_batch, mask_batch, resized_mask_batch, flipped_mask_batch]
    grid=0
    for row_idx in range(rows): 
        for col_idx in range(columns): 
            plt.subplot(gs[grid])
            plt.axis('off')
            plt.title(augmentation[col_idx%3])
            plt.imshow(grid_data[col_idx].at(row_idx))
            grid+=1
    plt.tight_layout()
```

```python
# DO NOT CHANGE THIS CELL
show_augmented_images(augmentation_pipe_output)
```

![data_augmentation](https://github.com/user-attachments/assets/113af07a-5fd8-4037-890e-a5c79c61bbc2)

