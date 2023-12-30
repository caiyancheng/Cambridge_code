###Author: Yancheng Cai

###Please use the following command to install the environment：

conda create -n ** python=3.9
pip install -r requirements.txt

###Specific description of the folder

"./TCSF": Contains two TCSF functions (standard TCSF curve and TCSF surface under different illuminance) and their visualization.

"./TCSF2_our_method_flicker_visibility": Added TCSF with variable illumination to my method.

"./VESA": VESA Method.

"./FMA": FMA Method.

"./JEITA": JEITA Method.

"./phase_flicker_visibility": Just a visualization showing why phase needs to be considered. A specific solution has not been thought out.

"./basic_flicker_visibility": No need to look it up, I misunderstood the 120Hz and 24Hz segments here.

"./pure_sum_flicker_visibility": Don't be misled by the name, this is the method I use.

"./example_in_book": Reproduced the examples in the book and found that there are many mistakes in the book, see supplementary materials.

