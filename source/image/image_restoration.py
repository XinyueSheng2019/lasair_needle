import sys
# sys.path.append('/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE_Image_Restoration/NEEDLE2.0/')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from quality_classification.quality_classification import QualityClassification
from utils import get_noise_distribution
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError





class ImageRestoration:
    """
    This class is used to restore the image of the transient object.
    """

    def __init__(self, obj_id, sci_data, sci_hdr, ref_data, ref_hdr, target_ra, target_dec, host_ra, host_dec, display = False):
        # object id
        self.obj_id = obj_id

        # science and reference images
        self.sci_data = sci_data
        self.sci_hdr = sci_hdr
        self.ref_data = ref_data
        self.ref_hdr = ref_hdr

        # target and host coordinates
        self.target_ra = target_ra
        self.target_dec = target_dec
        self.host_ra = host_ra
        self.host_dec = host_dec
        
        # check if the host exists
        self.hashost = self._has_host

        # map the coordinates to the pixel coordinates
        self.pixel_coords_target, self.pixel_coords_host = self._map_coordinate_with_pixel(is_sci=True, lasair = True)

        # padding the image
        self._padding_image(is_sci = True)
        self._padding_image(is_sci = False)

        # quality check model
        self.__quality_check_model = QualityClassification(verbose=False)
        self._display = display
 
    def _normalize_image(self, image):
        '''
        This function is used to normalize the image.
        '''   
        return (np.nanmax(image) - image) / (np.nanmax(image) - np.nanmin(image))

    def _check_shape(self, img):
        """
        Check if the image has the shape (60, 60) and does not contain any NaN values.
        """
        return img is not None and img.shape == (60, 60) and not np.any(np.isnan(img))

    def _border_mask(self, img, margin=3):
        """Return a boolean mask for the image border (margin pixels on each side)."""
        mask = np.zeros(img.shape, dtype=bool)
        if margin > 0:
            mask[:margin, :] = True
            mask[-margin:, :] = True
            mask[:, :margin] = True
            mask[:, -margin:] = True
        return mask

    def _has_border_nans(self, img, margin=3):
        """Check if any NaN values lie on the image border."""
        border = self._border_mask(img, margin)
        return np.any(np.isnan(img[border]))

    def _fill_border_nans(self, img, margin=3):
        """
        Fill border NaNs with the nanmedian of interior (good) pixels.
        Returns the filled image and whether any border NaNs were filled.
        """
        img = img.copy()
        border = self._border_mask(img, margin)
        border_nans = border & np.isnan(img)
        if not np.any(border_nans):
            return img, False

        interior = ~border
        good_pixels = img[interior & ~np.isnan(img)]
        if good_pixels.size == 0:
            good_pixels = img[~np.isnan(img)]
        if good_pixels.size == 0:
            return img, False

        fill_value = np.nanmedian(good_pixels)
        img[border_nans] = fill_value
        return img, True


    def quality_check(self, image):
        """
        check the quality of the image
        """
        image_copy = image.copy()
        if np.any(np.isnan(image_copy)):
            image_copy = np.nan_to_num(image_copy, nan = np.nanmean(image_copy))
            print('-------------------test and fill the nan with the mean value of the image: ', image_copy)
        if self._check_shape(image_copy):
            result = self.__quality_check_model.run(image_copy)
            return result
        else: 
            return 0
   
    def show_test_img(self, img, title_str):
        """
        show the test image
        """
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        result = self.quality_check(img.copy())
        plt.title(title_str + ' score: ' + str(result))
        plt.show()


    def _SSIM_restore(self, is_sci = True, threshold = 0.3):
        '''
        This function is used to restore the image of the transient object using SSIM method.
        It only works when two images have the same shape.
        If SSIM fails or border NaNs remain, fill border NaNs with interior median and re-check quality.
        '''

        def __match_contrast(image1, image2, replace_mask, mask1, mask2):
            filling_values = image2[replace_mask]
            if filling_values.size == 0:
                return None

            reference_mask = mask1 & mask2 & ~replace_mask
            image1_adjusted = image1[reference_mask]
            if image1_adjusted.size == 0:
                return None

            image1_std = np.nanstd(image1_adjusted)
            image2_ref = image2[mask2]
            image2_std = np.nanstd(image2_ref)

            if not np.isfinite(image2_std) or image2_std == 0:
                image2_std = 1e-5
            if not np.isfinite(image1_std) or image1_std == 0:
                image1_std = 1e-5

            scale_factor = image1_std / image2_std
            matched_value = (
                (filling_values - np.nanmean(image2_ref)) * scale_factor
                + np.nanmean(image1_adjusted)
            )

            if not np.all(np.isfinite(matched_value)):
                return None
            return matched_value

        def __update_restored_image(img):
            if is_sci:
                self.sci_data = img
            else:
                self.ref_data = img

        def __border_fallback(img):
            img, filled = self._fill_border_nans(img)
            if filled:
                __update_restored_image(img)
            return img

        def __quality_score(img):
            return self.__quality_check_model.run(img.copy())

        if is_sci:
            image1 = self.sci_data.copy()
            image2 = self.ref_data
        else:
            image1 = self.ref_data.copy()
            image2 = self.sci_data

        if image1.shape != image2.shape:
            print('the shape of the two images are not the same')
            return __quality_score(__border_fallback(image1))

        mask1 = ~np.isnan(image1)
        mask2 = ~np.isnan(image2)

        nor_image1 = self._normalize_image(image1)
        nor_image2 = self._normalize_image(image2)

        image1_filled = np.where(mask1, nor_image1, 255)
        image2_filled = np.where(mask2, nor_image2, 255)

        _, diff = ssim(image1_filled, image2_filled, data_range=255, full=True)

        diff = (1 - diff)  # Invert the SSIM map (highlight differences)

        # Threshold the difference map to find significant differences
        diff_mask = (diff > threshold).astype(np.uint8)  # Binary mask (0 or 1)
        if self._display:
            self.show_test_img(image1, 'image1')
            self.show_test_img(image2, 'image2')
            self.show_test_img(diff_mask, 'diff_mask')

        ssim_failed = np.sum(diff_mask) >= 1800
        if ssim_failed:
            print('the difference is too large, falling back to border median fill')
        else:
            replace_mask = (diff_mask == 1) & mask1 & mask2
            if np.any(replace_mask):
                matched_value = __match_contrast(image1, image2, replace_mask, mask1, mask2)
                if (
                    matched_value is not None
                    and matched_value.size == np.count_nonzero(replace_mask)
                ):
                    image1[replace_mask] = matched_value
                    __update_restored_image(image1)
                    if self._display:
                        self.show_test_img(image1, 'restored_image')
                else:
                    print('contrast match failed, falling back to border median fill')
                    ssim_failed = True
            else:
                print('no valid pixels for contrast match, falling back to border median fill')
                ssim_failed = True

        if ssim_failed or self._has_border_nans(image1):
            image1 = __border_fallback(image1)

        return __quality_score(image1)
   


  
    # def __check_host_region(self, img):
    #     # check if the host region is at the edge of the image
    #     if self.pixel_coords_host is None:
    #         return False
    #     r, l = int(self.pixel_coords_host[0]), int(self.pixel_coords_host[1])
    #     r = np.clip(r, 0, img.shape[0] - 1)
    #     l = np.clip(l, 0, img.shape[1] - 1)
    #     clipped = img - sigma_clip(img, sigma=3, maxiters=10)


    #     # if r == 0 or r == img.shape[0] - 1 or l == 0 or l == img.shape[1] - 1:
    #     #     return True
    #     return False


    def _padding_image(self, is_sci = True): 
        '''
        This function is used to padding the missing pixels with None.
        '''   
        def __adjust_length(x):
            return max(0, x)
            
        if is_sci:
            img = self.sci_data
        else:
            img = self.ref_data 
        
        # print('img.shape: ', img.shape)

        if img.shape == (60, 60):
            return

        if self.pixel_coords_target is None:
            return

        r, l = int(self.pixel_coords_target[0]), int(self.pixel_coords_target[1])
        
        # print('r, l: ', r, l)

        if r >= 0 and l >= 0:
            
            d1 = __adjust_length(30 - l)
            d2 = __adjust_length(30 - (img.shape[0]-l))
            d3 = __adjust_length(30 - r)
            d4 = __adjust_length(30 - (img.shape[1]-r))

            print('before d1, d2, d3, d4: ', d1, d2, d3, d4)
        
            if img.shape[0] == 60:
                d1 = 0
                d2 = 0
            if img.shape[1] == 60:
                d3 = 0
                d4 = 0

            print('d1, d2, d3, d4: ', d1, d2, d3, d4)
            img = np.pad(img, ((d1, d2), (d3, d4)), mode='constant', constant_values=None)
            
            # if self.__check_host_region(img):
            #     img = np.pad(img, ((d1, d2), (d3, d4)), mode='constant', constant_values=None) # type: ignore
            # else:
            #     noise = get_noise_distribution(img)
            #     # Create a new array with random values from the noise distribution
            #     new_shape = (img.shape[0] + d1 + d2, img.shape[1] + d3 + d4)
            #     padded_img = np.random.choice(noise, size=new_shape)
            #     # Place the original image in the center
            #     padded_img[d1:d1+img.shape[0], d3:d3+img.shape[1]] = img
            #     img = padded_img

            img = img[:60, :60]

            # update the pixel coordinates
            self.pixel_coords_target = [self.pixel_coords_target[0] + d3, self.pixel_coords_target[1] + d1]
            if self.pixel_coords_host is not None:
                self.pixel_coords_host = [self.pixel_coords_host[0] + d3, self.pixel_coords_host[1] + d1]


            if is_sci:
                self.sci_data = img
            else:
                self.ref_data = img

        return 
            
    def _create_wcs_lasair(self, shape, ra_deg, dec_deg, scale_arcsec=1.0):
        """Create WCS for Lasair cutout: center at (ra, dec), scale_arcsec per pixel."""
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [ra_deg, dec_deg]
        wcs.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
        dec_rad = np.radians(dec_deg)
        deg_per_arcsec = 1.0 / 3600.0
        wcs.wcs.cdelt = [-deg_per_arcsec / np.cos(dec_rad), deg_per_arcsec]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.array_shape = shape
        return wcs

    def _cutout_img(self, data, header, ra, dec, size=60):
        # cutout the image to proposed size
        try:
            pixels = WCS(header=header).all_world2pix(ra, dec, 1)
            pixels = [int(x) for x in pixels]
            cutout = Cutout2D(data, position=pixels, size=size)
            return cutout.data, cutout.wcs
        except (NoOverlapError, Exception):
            # Fallback: Lasair cutouts lack full WCS; build one with center at target, 1"/pixel
            center = (data.shape[1] / 2.0, data.shape[0] / 2.0)
            cutout = Cutout2D(data, position=center, size=size)
            wcs = self._create_wcs_lasair(cutout.data.shape, ra, dec, scale_arcsec=1.0)
            return cutout.data, wcs
       
    @property
    def _has_host(self):
        if self.host_ra is None and self.host_dec is None:
            hashost = False
        else:
            hashost = True 
        return hashost
            

    def _map_coordinate_with_pixel(self, is_sci, lasair = False): 


        if is_sci:
            hdr = self.sci_hdr
        else:
            hdr = self.ref_hdr
        
        if lasair:
            # Lasair cutouts: always use custom WCS (center at target, 1"/pixel); header WCS is often incomplete
            data, _ = self._cutout_img(self.sci_data, hdr, self.target_ra, self.target_dec)
            wcs = self._create_wcs_lasair(data.shape, self.target_ra, self.target_dec, scale_arcsec=1.0)
            pixel_coords_target = [data.shape[0] // 2, data.shape[1] // 2] 
            # pixel_coords_target = [30, 30]
            if self.host_ra is not None and self.host_dec is not None:
                print('-------------------test host_ra, host_dec: ', self.host_ra, self.host_dec)
                print('-------------------test target_ra, target_dec: ', self.target_ra, self.target_dec)
                pixel_coords_host_raw = SkyCoord(ra=self.host_ra, dec=self.host_dec, unit="deg").to_pixel(wcs)
                pixel_coords_host = [int(pixel_coords_host_raw[0]), int(pixel_coords_host_raw[1])]
            else:
                pixel_coords_host = None

            return pixel_coords_target, pixel_coords_host
        else:
        
            try: 
                wcs = WCS(hdr)
                # # Convert RA and DEC to pixel coordinates, find the target pixels
                target_coords = SkyCoord(ra=self.target_ra, dec=self.target_dec, unit="deg")
                pixel_coords_target_raw = target_coords.to_pixel(wcs)
                # print('------------- test map coordinate with pixel: pixel_coords_target_raw: ', pixel_coords_target_raw)
                pixel_coords_target = [float(pixel_coords_target_raw[0]), float(60 - pixel_coords_target_raw[1])]
            
                # Convert RA and DEC to pixel coordinates, find the host pixels
                if self.host_ra is not None and self.host_dec is not None:
                    host_coords = SkyCoord(ra=self.host_ra, dec=self.host_dec, unit="deg")
                    pixel_coords_host = host_coords.to_pixel(wcs)
                    if is_sci:
                        pixel_coords_host = [float(pixel_coords_host[0]), float(60 - pixel_coords_host[1])]
                    else:
                        pixel_coords_host = [float(60 - pixel_coords_host[0]), float(60 - pixel_coords_host[1])]
                else:
                    pixel_coords_host = None

                return pixel_coords_target, pixel_coords_host
            except: 
       
                return None, None
           

        


#--------------------------------
# Load bad samples (local files for testing)
#--------------------------------

def load_bad_samples():
    f = open('Bad_Samples.json', 'r')
    bad_samples = json.load(f)
    return bad_samples['sci_data_bogus']


def load_samples():
    f = pd.read_csv('/Users/xinyuesheng/Documents/astro_projects/data/ztf_transients_all.csv')
    return f['ZTFID'].tolist()
