---
triggers:
  - PPT
  - pptx
  - Excel
  - xlsx
  - PDF
  - 文档
  - 表格
  - 幻灯片
  - 演示文稿
  - 报告
  - 简历
  - resume
  - spreadsheet
  - 做个
  - 生成
max_tokens: 1200
lock_session: false
---

# 生成文件（PPT/Excel/PDF/Word）

## 效率要求（最重要）

**流程：**
1. **先回复用户**：简要说明你打算做什么（几页、什么内容、预计需要的时间），让用户知道你在处理
2. `browser` → 搜索并下载与主题相关的图片，数量自行判断
3. `file_write` → 写完整 Python 脚本到 `/tmp/gen_xxx.py`（脚本中用 `add_picture()` 插入已下载的图片）
4. `bash` → 执行脚本 `./python/bin/python3.12 /tmp/gen_xxx.py`
5. 告诉用户文件路径

**严禁：**
- 不要用 `python -c '...'` 或 `python3 -c '...'` — 脚本太长会截断
- 不要写完脚本后又 file_edit 修改 — 一次写对
- 不要执行失败后反复修改重试 — 检查好再执行
- 不要分多次 file_write — 一次写完整个脚本
- **图片路径必须在脚本中硬编码为绝对路径字符串**，例如 `img1 = "/Users/.../image_abc.jpg"`。严禁通过 `os.environ.get()`、命令行参数、外部配置文件等方式传入图片路径——执行环境不保证设置了这些变量，会导致图片全部丢失

## PPT (python-pptx) 规范

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
```

每页必须有具体内容，不是只有标题。

配色推荐：主色 `RGBColor(0x2B, 0x57, 0x9A)`，辅色 `RGBColor(0x5B, 0x9B, 0xD5)`。

页面类型：
1. **封面页**: 大标题 + 副标题 + 色条装饰
2. **内容页**: 顶部色条标题 + 正文要点（每个要点有 1-3 句具体描述）
3. **表格页**（可选）: `slide.shapes.add_table()` 填入真实数据
4. **总结页**: 要点回顾

字号：标题 28-36pt 加粗，正文 16-20pt。

内容要求：
```
❌ 错误: "第一天：鼓浪屿风情"
✅ 正确: "第一天：鼓浪屿风情
   上午: 渡轮前往（35元/人），游日光岩（60元）→ 菽庄花园（30元）
   午餐: 龙头路小吃街（海蛎煎、沙茶面）
   预算: 约 200-300 元/人"
```

**图片处理（重要，必须遵守）：**
- 先用 `browser` 工具搜索并下载与主题相关的图片
- 图片路径必须使用 browser 工具返回的真实路径，严禁编造
- 如果用户也上传了图片，优先使用用户的图片
- **严禁直接 `add_picture(path, left, top, width, height)` 同时指定宽高——会变形拉伸**
- **所有图片插入必须使用下面的 `add_picture_cropped` 函数**（直接复制到脚本开头）：

```python
from PIL import Image as PILImage

def add_picture_cropped(slide, img_path, left, top, target_w, target_h):
    """插入图片到指定区域：等比缩放 + 居中裁剪，不变形。"""
    with PILImage.open(img_path) as im:
        iw, ih = im.size
    img_ratio = iw / ih
    box_ratio = target_w / target_h
    if img_ratio > box_ratio:
        scale_h = target_h
        scale_w = target_h * img_ratio
    else:
        scale_w = target_w
        scale_h = target_w / img_ratio
    pic = slide.shapes.add_picture(
        img_path,
        int(left - (scale_w - target_w) / 2),
        int(top - (scale_h - target_h) / 2),
        int(scale_w), int(scale_h),
    )
    crop_lr = (scale_w - target_w) / 2 / scale_w
    crop_tb = (scale_h - target_h) / 2 / scale_h
    pic.crop_left = crop_lr
    pic.crop_right = crop_lr
    pic.crop_top = crop_tb
    pic.crop_bottom = crop_tb
    pic.left = int(left)
    pic.top = int(top)
    pic.width = int(target_w)
    pic.height = int(target_h)
```

调用示例：`add_picture_cropped(slide, "/tmp/photo.jpg", Inches(1), Inches(1.5), Inches(5), Inches(4))`
**封面底图遮挡（必须遵守）：**
- 禁止用全屏纯色矩形覆盖底图（尤其是黑色块）
- 背景色只能用 `slide.background.fill` 设置
- 若需要提升文字可读性，只允许“小面积半透明条/阴影”，且必须在图片之后添加并保持可见底图

## Word (python-docx) 规范

- Heading 样式分层级，段落有缩进
- **图片插入必须保持比例，不可变形**。用 PIL 预处理后再插入：

```python
from docx.shared import Inches
from PIL import Image

def add_image_cover(doc_or_para, img_path, target_w_in, target_h_in):
    """Word 插图：等比缩放 + PIL 居中裁剪，不变形。"""
    with Image.open(img_path) as im:
        iw, ih = im.size
        img_ratio = iw / ih
        box_ratio = target_w_in / target_h_in
        if img_ratio > box_ratio:
            new_h = ih
            new_w = int(ih * box_ratio)
        else:
            new_w = iw
            new_h = int(iw / box_ratio)
        left = (iw - new_w) // 2
        top = (ih - new_h) // 2
        cropped = im.crop((left, top, left + new_w, top + new_h))
        cropped.save(img_path)  # 覆盖原文件或另存
    run = doc_or_para.add_run() if hasattr(doc_or_para, 'add_run') else doc_or_para.paragraphs[-1].add_run()
    run.add_picture(img_path, width=Inches(target_w_in))
```

## HTML 图片规范

HTML 中图片不变形的正确方式是用 CSS `object-fit: cover`，**严禁同时写死 width+height 不加 object-fit**：

```html
<!-- ✅ 正确：填满容器 + 居中裁剪 -->
<div style="width:400px; height:300px; overflow:hidden;">
  <img src="photo.jpg" style="width:100%; height:100%; object-fit:cover;">
</div>

<!-- ❌ 错误：硬拉变形 -->
<img src="photo.jpg" width="400" height="300">
```

若用 Python 生成 HTML，图片标签必须包含 `object-fit:cover` 和容器约束。

## Excel / PDF

- Excel: 首行冻结+加粗+背景色，列宽自适应，数据有边框；插图同样用 PIL 先裁剪再 `ws.add_image()`
- PDF (reportlab): 注册中文字体（PingFang.ttc），合理边距；插图用 PIL 预裁剪到目标比例后再 `drawImage()`

## 关键提醒

- 脚本必须完整可运行，写之前在脑中过一遍有没有语法错误
- 不需要安装依赖（已预装 python-pptx / openpyxl / reportlab / python-docx）
- 保存到 `/tmp/<有意义的文件名>.<后缀>`
- 文件名必须去除空格与特殊符号，仅保留中文/英文/数字/下划线，例如：`/tmp/同安2日游.pptx`
- 用户说"做个 PPT"= 完整可用的文件，不是骨架
