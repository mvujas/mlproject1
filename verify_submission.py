def verify(*filenames):
  num_files = len(filenames)
  lines = []
  for filename in filenames:
    with open(filename, 'r') as f:
      lines.append(f.readlines())
  
  all_same = True
  i = 0
  while all_same and i < num_files - 1:
    l1 = lines[i]
    l2 = lines[i  + 1]
    all_same &= (len(l1) == len(l2))
    if all_same:
      num_lines = len(l1)
      k = 0
      while all_same and k < num_lines:
        all_same &= (l1[k] == l2[k])
        k += 1
    i += 1
  return all_same

files_same = verify('submission.csv', 'submission_123.csv')

print('Files ' + ('are' if files_same else 'are not') + ' same')