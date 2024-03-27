# code that was part of the app, that could still be useful

# def create_class_selector(self):
#     with st.expander("Select objects to be detected"):
#         classes_df = self.load_classes()
#         classes_df["enabled"] = False

#         self.columns = list(classes_df.columns)

#         self.categories = list(classes_df["category"].unique())

#         self.filtered_df = {}

#         for i, category in enumerate(self.categories):
#             self.filtered_df[category] = classes_df[
#                 classes_df["category"] == category
#             ]
#             self.filtered_df[category] = st.data_editor(
#                 self.filtered_df[category],
#                 column_config={
#                     "enabled": st.column_config.CheckboxColumn(
#                         "enabled",
#                         help="Select the classes you want the model to detect",
#                     )
#                 },
#                 disabled=list(classes_df.columns[:-1]),
#                 hide_index=True,
#                 key=category,
#             )

# def return_selected_classes(self):

#     total_df = pd.DataFrame(columns=self.columns)

#     for category in self.categories:

#         temp = self.filtered_df[category][
#             self.filtered_df[category]["enabled"] == True
#         ]
#         total_df = pd.concat([total_df, temp])
#     self.selected_classes = list(total_df.index.values)

# def initialize_session_state(self):
#     if "clicked" not in st.session_state:
#         st.session_state.clicked = True
