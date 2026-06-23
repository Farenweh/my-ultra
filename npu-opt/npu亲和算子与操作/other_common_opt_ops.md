
# 其他常用优化操作

- 配置ACLNN\_CACHE\_LIMIT环境变量

    ACLNN\_CACHE\_LIMIT用于配置单算子在执行API时，在Host侧缓存的算子信息条目个数。ACLNN\_CACHE\_LIMIT的单位为个，取值范围为\[1,10000000\]，默认值为10000。缓存的算子信息包含workspace大小、算子计算的执行器以及tiling信息等。动态shape场景下，如算子shape范围较大时，用户可通过配置此环境变量适当增加算子缓存条目，提升调度性能。当模型遇到下发瓶颈时（尤其是问题出现在一级流水时），可以尝试此配置。配置方法如下：

    ```shell
    export ACLNN_CACHE_LIMIT=100000
    ```

    > [!NOTE]
    >
    > 增加算子信息缓存条目会增加Host内存开销，需根据实际情况适当调整。

- 非连续转连续消除

    由于底层实现的差异，GPU上支持对非连续tensor进行操作不会产生额外性能问题，而对于NPU则需要插入非连续转连续算子。因此GPU上某些代码运行会连续对一个非连续tensor进行操作，对NPU来说就会插入多余的转换算子造成性能下降。用户可以提前将非连续的tensor转连续，避免后续的额外转换算子插入。

    通常，permute和transpose操作都会造成tensor非连续。示例如下：

    ```python
    # 非连续情况
    x = input_.permute(0, 1)  
    y1 = norm1(x)
    y2 = norm2(x)
    ```

    在上述示例中，x由于permute操作而变得非连续，因此在norm1和norm2的计算中，会额外进行一次操作，用于保证x的连续。通过调用contiguous方法，可以把原来的两次非连续操作变为一次，从而得到性能的提升。示例如下：

    ```python
    # 非连续转连续
    x = input_.permute(0, 1).contiguous() 
    y1 = norm1(x)
    y2 = norm2(x)
    ```

- 换轴reduce，换轴index

    由于NPU侧有些算子的泛化性能较差，因此对于一些非尾轴reduce的算子以及一些非首轴的随机访存算子，优化有显著问题。此时可以先transpose进行尾轴reduce或首轴随机访存，再transpose回来，也许性能将得到提升。

    例如，对一个三轴的tensor x，shape为（dim1, dim2, dim3），如果发现会对dim2进行高频操作，可以先对x进行transpose，变为（dim2, dim1, dim3），完成操作后再恢复为原来的shape，保证内存的高效搬运。

- mmcv的optimizer加入runner \(限昇腾适配的mmcv的optimizer\)，会有性能收益。
- 使用NPU-Fast\_GELU

    GELU实现公式有很多，大致可以分为以下三个版本：

    - Google-GELU，是PyTorch官方默认版本，NPU上没有对应的实现，这种实现性能也是最慢的（未证实）。

        公式：![](./figures/other_common_opt_ops_fig_01.png)

    - OpenAI-GELU，也可以称作GPU的Fast\_GELU，这个实现是Google-GELU的近似版本，PyTorch 2.0之后已支持，通过设置approximate='tanh'使能，实现比Google-GELU更快。当前NPU上默认是用这种实现的，所以说NPU的GELU等价于GPU的Fast\_GELU，这种实现已经用于GPT训练，因此可以看作是业界主流。

        公式：![](./figures/other_common_opt_ops_fig_02.png)

    - NPU-Fast\_GELU，NPU的Fast\_GELU是在OpenAI-GELU也就是GPU的Fast\_GELU公式基础上，通过泰勒展开逼近得到的新公式，性能更快，同时在GPU上并没有对应的使用，有可能产生精度问题。
