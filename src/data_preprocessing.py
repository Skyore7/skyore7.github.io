import pandas as pd
import numpy as np
import torch
import h5py
from sklearn.preprocessing import StandardScaler

# Example of reading large CSV in chunks
chunk_size = 10000  # 10k rows per chunk (adjust according to memory)

def preprocessTrain(filename):
    h5_filename = filename + ".h5"
    with h5py.File(h5_filename, "w") as f:
        for chunk in pd.read_csv(filename + ".csv", chunksize=chunk_size):

            chunk = chunk.groupby('event_id').apply(lambda x: x.sample(n=min(len(x), 10), random_state=42)).reset_index(drop=True)

            # Preprocess the chunk
            chunk.drop(columns=['game_num', 'event_id', 'event_time', 'player_scoring_next', 'team_scoring_next', 
                                'boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer', 'boost4_timer', 'boost5_timer'])

            # Iterate through each player (p0 to p5)
            for i in range(6):
                boost_col = f'p{i}_boost'

                # Create the 'demoed' column for the player
                chunk[f'p{i}_demoed'] = np.where(chunk[boost_col].isna(), 1, 0)

                # Check if boost is NaN for the current player and adjust their attributes
                if chunk[boost_col].isna().any():
                    chunk[f'p{i}_pos_x'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_pos_x'])
                    chunk[f'p{i}_pos_y'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_pos_y'])
                    chunk[f'p{i}_pos_z'] = np.where(chunk[boost_col].isna(), 10000, chunk[f'p{i}_pos_z'])
                    chunk[f'p{i}_vel_x'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_vel_x'])
                    chunk[f'p{i}_vel_y'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_vel_y'])
                    chunk[f'p{i}_vel_z'] = np.where(chunk[boost_col].isna(), 10000, chunk[f'p{i}_vel_z'])
                    chunk[boost_col] = np.where(chunk[boost_col].isna(), 0, chunk[boost_col])
            # Add the 'no_team_scoring_within_10sec' column
            chunk['no_team_scoring_within_10sec'] = np.where(
                (chunk['team_A_scoring_within_10sec'] == 0) & (chunk['team_B_scoring_within_10sec'] == 0),
                1, 0
            )

            # Reorganize columns to the desired order:
            ball_columns = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z']
            player_columns = []
            for i in range(6):
                player_columns += [f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z',
                                   f'p{i}_vel_x', f'p{i}_vel_y', f'p{i}_vel_z', f'p{i}_boost', f'p{i}_demoed']

            scoring_columns = ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec', 'no_team_scoring_within_10sec']

            # Combine all columns in the desired order
            final_columns = ball_columns + player_columns + scoring_columns
            chunk = chunk[final_columns]

            # Convert the chunk to a tensor
            data_tensor = torch.tensor(chunk.values, dtype=torch.float32)

            # Check if the dataset already exists and create or append accordingly
            if "data" not in f:
                # Create dataset if it doesn't exist
                f.create_dataset("data", data=data_tensor.numpy(), maxshape=(None, data_tensor.shape[1]), chunks=True, compression="gzip")
            else:
                # Append data to the existing dataset
                existing_data = f["data"]
                current_shape = existing_data.shape
                new_shape = (current_shape[0] + data_tensor.shape[0], current_shape[1])
                
                # Resize the dataset to accommodate the new data
                existing_data.resize(new_shape)

                # Append the new data
                existing_data[current_shape[0]:, :] = data_tensor.numpy()

    return h5_filename

def preprocessTest(filename):
    h5_filename = filename + ".h5"
    with h5py.File(h5_filename, "w") as f:
        for chunk in pd.read_csv(filename + ".csv", chunksize=chunk_size):
            # Preprocess the chunk
            chunk.drop(columns=['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer', 'boost4_timer', 'boost5_timer'])

            # Iterate through each player (p0 to p5)
            for i in range(6):
                boost_col = f'p{i}_boost'

                # Create the 'demoed' column for the player
                chunk[f'p{i}_demoed'] = np.where(chunk[boost_col].isna(), 1, 0)

                # Check if boost is NaN for the current player and adjust their attributes
                if chunk[boost_col].isna().any():
                    chunk[f'p{i}_pos_x'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_pos_x'])
                    chunk[f'p{i}_pos_y'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_pos_y'])
                    chunk[f'p{i}_pos_z'] = np.where(chunk[boost_col].isna(), 10000, chunk[f'p{i}_pos_z'])
                    chunk[f'p{i}_vel_x'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_vel_x'])
                    chunk[f'p{i}_vel_y'] = np.where(chunk[boost_col].isna(), 0, chunk[f'p{i}_vel_y'])
                    chunk[f'p{i}_vel_z'] = np.where(chunk[boost_col].isna(), 10000, chunk[f'p{i}_vel_z'])
                    chunk[boost_col] = np.where(chunk[boost_col].isna(), 0, chunk[boost_col])

            # Reorganize columns to the desired order:
            id_column = ['id']
            ball_columns = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z']
            player_columns = []
            for i in range(6):
                player_columns += [f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z',
                                   f'p{i}_vel_x', f'p{i}_vel_y', f'p{i}_vel_z', f'p{i}_boost', f'p{i}_demoed']

            # Combine all columns in the desired order
            final_columns = id_column + ball_columns + player_columns
            chunk = chunk[final_columns]

            # Convert the chunk to a tensor
            data_tensor = torch.tensor(chunk.values, dtype=torch.float32)

            # Check if the dataset already exists and create or append accordingly
            if "data" not in f:
                # Create dataset if it doesn't exist
                f.create_dataset("data", data=data_tensor.numpy(), maxshape=(None, data_tensor.shape[1]), chunks=True, compression="gzip")
            else:
                # Append data to the existing dataset
                existing_data = f["data"]
                current_shape = existing_data.shape
                new_shape = (current_shape[0] + data_tensor.shape[0], current_shape[1])
                
                # Resize the dataset to accommodate the new data
                existing_data.resize(new_shape)

                # Append the new data
                existing_data[current_shape[0]:, :] = data_tensor.numpy()

    return h5_filename
