## Guided Filtering Layer
### Install
* PyTorch Version
    ```sh
    pip install -e GuidedFilteringLayer/GuidedFilter_PyTorch
    ```
* Tensorflow Version
    ```sh
    pip install -e GuidedFilteringLayer/GuidedFilter_TF
    ```

### Usage
* PyTorch Version
    ```python
    from guided_filter_pytorch.guided_filter import FastGuidedFilter
    
    hr_y = FastGuidedFilter(r, eps)(lr_x, lr_y, hr_x)
    ```
    ```python
    from guided_filter_pytorch.guided_filter import GuidedFilter
    
    hr_y = GuidedFilter(r, eps)(hr_x, init_hr_y)
    ``` 
* Tensorflow Version
    ```python
    from guided_filter_tf.guided_filter import fast_guided_filter
    
    hr_y = fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)
    ```
    ```python
    from guided_filter_tf.guided_filter import guided_filter
    
    hr_y = guided_filter(hr_x, init_hr_y, r, eps, nhwc)
    ``` 