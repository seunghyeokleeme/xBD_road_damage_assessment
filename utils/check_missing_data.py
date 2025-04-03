import os
import pandas as pd

lst_test_change = [file.split('_label3class.png')[0] for file in os.listdir('./xbd/test/test-change') if file.endswith('.png')]
lst_test_post = [file.split('_post_disaster.png')[0] for file in os.listdir('./xbd/test/test-post') if file.endswith('.png')]
lst_test_pre = [file.split('_pre_disaster.png')[0] for file in os.listdir('./xbd/test/images') if file.endswith('.png')]

lst_test_change.sort()
lst_test_post.sort()
lst_test_pre.sort()

print("test-change:", len(lst_test_change))
print("test-post:", len(lst_test_post))
print("test-pre:", len(lst_test_pre))

print("================================================")
pre_missing_files = [file for file in lst_test_post if file not in lst_test_pre]
post_missing_files = [file for file in lst_test_pre if file not in lst_test_post]
change_missing_files = [file for file in lst_test_change if file not in lst_test_pre]

print("pre_missing_files:",  len(pre_missing_files))
print("post_missing_files:", len(post_missing_files))
print("change_missing_files:", len(change_missing_files))

missing_pre_df = pd.DataFrame({'test_pre': pre_missing_files})
missing_pre_df.to_csv('xbd_missing_pre.csv', index=False)

missing_post_df = pd.DataFrame({'test_post': post_missing_files})
missing_post_df.to_csv('xbd_missing_post.csv', index=False)
