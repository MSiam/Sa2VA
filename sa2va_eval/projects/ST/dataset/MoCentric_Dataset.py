import numpy as np
import copy
from PIL import Image
import torch
import os
import json

from .encode_fn import video_lisa_encode_fn
from .ReVOS_Dataset import VideoReVOSDataset


class VideoMoCentricDataset(VideoReVOSDataset):

    def __init__(self, **kwargs):
        self.selected_frame_type = kwargs['selected_frame_type']
        selected_frames_file = kwargs['selected_frames_file']
        fix_aspect_ratio = kwargs['fix_aspect_ratio']
        self.left_flag_type = kwargs['left_flag_type']
        self.single_only = kwargs['single_only']

        del kwargs['selected_frame_type']
        del kwargs['selected_frames_file']
        del kwargs['fix_aspect_ratio']
        del kwargs['left_flag_type']
        del kwargs['single_only']

        selected_frames = {}
        with open(selected_frames_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                video_exp = data['video_name'] + '_' + data['exp_id']
                selected_frames[video_exp] = data['frame']

        self.selected_frames = selected_frames
        super().__init__(**kwargs)
        if fix_aspect_ratio:
            self.transformer.transforms[1].size = (448,)
#        self.extra_image_processor.both_sides = False

    def __getitem__(self, index):
        index = index % self.real_len()
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        # Position flag for the static frame
        if self.left_flag_type == 'none':
            left_flag = False
        elif self.left_flag_type == 'random':
            left_flag = bool(np.random.randint(2))

        if self.single_only == 'none':
            single_flag_only = False
        elif self.single_only == 'random':
            single_flag_only = bool(np.random.randint(2))

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        first_img_file = os.path.join(self.image_folder, video_objects_infos[0]['video'],
                                      video_objects_infos[0]['frames'][0] + '.jpg')
        first_img = Image.open(first_img_file)
        first_img = np.concatenate([np.array(first_img), np.array(first_img)], axis=1)
        first_img = Image.fromarray(first_img)
        first_img_size_postproc = self.transformer(first_img).shape
        patch_size = 14
        self.patch_token = (first_img_size_postproc[1] // patch_size, first_img_size_postproc[2] // patch_size)
        self.patch_token = (self.patch_token[0] - self.patch_token[0]%2, self.patch_token[1] - self.patch_token[1]%2)
        self.patch_token = (int(self.patch_token[0] * self.downsample_ratio), int(self.patch_token[1] * self.downsample_ratio))
        self.patch_token = self.patch_token[0] * self.patch_token[1]
        data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames)

        assert 'images' in data_dict.keys()
        pixel_values = []
        extra_pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None
        if data_dict.get('images', None) is not None:
            frames_files = data_dict['images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]

            # Select Static Image randomly
            if self.selected_frame_type == 'random':
                static_frame_path = np.random.choice(frames_files, 1)[0]
            elif self.selected_frame_type == 'keyframe':
                try:
                    vid_eid = video_objects_infos[0]['video'] + '_' + video_objects_infos[0]['exp_id']
                    static_frame_path = self.selected_frames[vid_eid]
                except:
                    # In case the video expression pair doesnt have a keyframe
                    static_frame_path = np.random.choice(frames_files, 1)[0]

            static_frame_image = Image.open(static_frame_path).convert('RGB')

            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')

                if single_flag_only:
                    cat_image = np.array(static_frame_image)
                else:
                    if left_flag:
                        cat_image = np.concatenate([np.array(static_frame_image), np.array(frame_image)], axis=1)
                    else:
                        cat_image = np.concatenate([np.array(frame_image), np.array(static_frame_image)], axis=1)

                ori_width, ori_height = frame_image.size
                frame_image = Image.fromarray(cat_image)
                if self.extra_image_processor is not None:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)

                if self.preprocessor is not None:
                    pass
                else:
                    frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    num_frames = _data_dict['image_grid_thw'].shape[0]
                    num_video_tokens = num_frame_tokens * num_frames
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                else:
                    raise NotImplementedError
                data_dict.update(_data_dict)
            else:
                pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
                data_dict['pixel_values'] = pixel_values
            if self.extra_image_processor is not None:
                data_dict['g_pixel_values'] = extra_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['video_masks'], image_size=(ori_height, ori_width))
            cat_masks = []
            for mask in masks:
                if single_flag_only:
                    mask = np.zeros(mask.shape)
                else:
                    if left_flag:
                        mask = np.concatenate([np.zeros(mask.shape), mask], axis=1)
                    else:
                        mask = np.concatenate([mask, np.zeros(mask.shape)], axis=1)
                cat_masks.append(mask)
            masks = cat_masks
            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))
            data_dict['masks'] = masks
        else:
            data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            data_dict['masks'] = None

        if num_video_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            frames_files = data_dict['fast_images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['fast_video_masks'], image_size=(ori_height, ori_width))

            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))

            data_dict['fast_exists'] = masks.to(dtype=torch.int).sum(dim=(-2, -1)).ge(self.exist_thr).unsqueeze(-1)


            del data_dict['fast_video_masks']
        data_dict['type'] = 'video'
        return data_dict


