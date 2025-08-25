# touch /tmp/init_marker
# proj_dir=$(pwd)

exp="11M_omnieditSFT"
step="00250"

dataname=mse_bench
root_dataset=multi_turn

save_dir=output/$dataname/${exp}_ckpt${step}_open

mkdir -p ckpt/VINCIE-3B; cd ckpt/VINCIE-3B
hdfs dfs get hdfs://harunasg/home/byte_seed_vgfm/intern/leigang.qu/exp/vfm_3b/session_editing_v1_sg/11M_omnieditSFT/states/0000000250/models/dit.pth
hdfs dfs get hdfs://harunava/home/byte_lab_video_va/user/leigang.qu/ckpt/vfm_3b/v.pth
mv v.pth vae.pth
hdfs dfs get hdfs://harunava/home/byte_lab_video_va/user/leigang.qu/ckpt/vfm_3b/t.tar.gz
tar -xzvf t.tar.gz

# cd ../../benchmark
# mkdir mse_bench; cd mse_bench
# hdfs dfs get hdfs://harunasg/home/byte_seed_vgfm/intern/leigang.qu/dataset/session_editing/bench/benchV4iter_combined/benchV4iter_coco_laion_en_aes_seed42_turn5_combinedV4_sampled_n100.zip
# unzip benchV4iter_coco_laion_en_aes_seed42_turn5_combinedV4_sampled_n100.zip
# mkdir tmp_data; cd tmp_data; mkdir iter_res_data_vinci10_promptV2_n184; cd iter_res_data_vinci10_promptV2_n184
# hdfs dfs get hdfs://harunasg/home/byte_seed_vgfm/intern/leigang.qu/dataset/session_editing/bench/paper/iter_res_data_vinci10_promptV2_n184.zip
# unzip iter_res_data_vinci10_promptV2_n184.zip


# ./main.sh configs/generate_distributed.yaml \
#     generation.positive_prompt.path=benchmark/mse_bench \
#     generation.output.dir=$save_dir \
#     generation.batch_size=16 \
#     dit.checkpoint=ckpt/VINCIE-3B/dit.pth \
#     vae.checkpoint=ckpt/VINCIE-3B/vae.pth


# generation.positive_prompt.path=benchmark/mse_bench \
# generation.output.dir=$save_dir \

# turn1="Remove the hat from Leonardo's head to reveal more of his hair. "
# turn2="Replace Leonardo's dark garment with a colorful and intricately patterned cloak. "
# turn3="Change the background to a richly detailed Renaissance-style room. "
# turn4="Alter the color of Leonardo's beard to a more youthful darker shade. "
# turn5="Enhance the lighting to softly illuminate Leonardo's features, creating a dynamic effect."




# turn1="Add modern eyeglasses to the portrait. "
# turn2="Change the background to a colorful renaissance-style fresco. "
# turn3="Modify the attire to a modern formal suit. "
# turn4="Replace the dark hat with a light-colored fedora. "
# turn5="Add a smartphone in the figure's hand. "

# python main.py configs/generate.yaml \
#     generation.positive_prompt.seed=1024 \
#     generation.positive_prompt.image_path=assets/da_vinci.png \
#     generation.positive_prompt.prompts="[\"$turn1\", \"$turn2\", \"$turn3\", \"$turn4\", \"$turn5\"]"


# turn1="Add colorful butterflies in the hair area."
# turn2="Add stylish modern glasses."
# turn3="Change the hair color to a soft blue tint."
# turn4="Replace the background with an intricate pattern of gears and mechanisms."
# turn5="Transform the image into a cubist painting."
# input_img=assets/da_vinci_1.png

# turn1="Add color to the hat."
# turn2="Replace the background with a Renaissance landscape, rendering the character more tangible to harmonize with the overall figure."
# turn3="Change the hat to a cowboy hat."
# turn4="Have him hold a paintbrush."
# turn5="Change the expression to a cheerful smile."
# input_img=assets/da_vinci_2.png
# output_dir=output/da_vinci_2

# turn1="Add a small dog playing near the girl."
# turn2="Change the color of the girl's coat to teal."
# turn3="Replace the snowy background with a spring meadow."
# turn4="Move the fire hydrant slightly to the right."
# turn5="Have the girl squat down."
# input_img=assets/girl.png
# output_dir=output/girl_camera2

turn1="Change the color of the puppy to white, but keep the pink clothes and the hat unchanged."
turn2="Add a small black kitten sitting beside the puppy in the lower right corner."
turn3="Change the background to an indoor setting."
turn4="Make the dog open its mouth and bark loudly."
turn5="Change the image into a watercolor painting."
input_img=assets/dog.png
output_dir=output/dog

# turn1="Change the man’s posture so that he lowers his hand and clasps both hands in front of him."
# turn2="Change the man’s expression to a broad, warm smile."
# turn3="Add a woman in formal attire standing slightly behind him to the right."
# turn4="Change the background to a colorful abstract graphic."
# turn5="Add a Superman cape to the man."
# turn1="Adjust the man's posture to a slight forward lean."
# turn2="Add a microphone to the man's right hand."
# turn3="Move the person slightly closer to the man."
# turn4="Replace the background with a garden environment."
# turn5="Change the man's expression to a warm smile."
# input_img=assets/man.png
# output_dir=output/man1


# turn1="Add a group of ducks swimming around the tree reflections."
# turn2="Replace the background with a moonlit sky to enhance the night-time atmosphere."
# turn3="It begins to snow."
# turn4="Freeze the lake surface, and add a person ice-skating."
# turn5="Position a reindeer near the red tree with its orientation towards the lights as if admiring them."
# input_img=assets/night.png
# output_dir=output/night


# export CAMERA_PREFIX="[###CAMERA: None###] "
# export CAMERA_PREFIX="Keep the camera unchanged. [###CAMERA: None###] "

# export CUDA_VISIBLE_DEVICES="0"
# python main.py configs/generate.yaml \
#     generation.positive_prompt.image_path=$input_img \
#     generation.positive_prompt.prompts="[\"$turn1\", \"$turn2\", \"$turn3\", \"$turn4\", \"$turn5\"]" \
#     generation.output.dir=$output_dir



# 
# 
# 
# 
# 