-- query for training data

WITH all_events AS ( -- any event we could be interested in at any given point
    select *
    from raw.src_snowplow.events
    WHERE UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1 IS NOT NULL
        AND derived_tstamp >= dateadd(day, -41, current_date-1)
        AND derived_tstamp < current_date-1
),
eligible_events AS ( -- The window of events where we will check for target
    select *
    from all_events
    WHERE UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1 IS NOT NULL
        AND derived_tstamp >= dateadd(day, -15, current_date-1)
        AND derived_tstamp < dateadd(day, -8, current_date-1)
),
events_with_target AS ( -- run through each event and check if target is true for event
    select eligible_events.event_id,
    EXISTS (SELECT 1 
        FROM all_events
        WHERE user_id = eligible_events.user_id 
            AND derived_tstamp <= DATEADD(day, 1, eligible_events.derived_tstamp)
            AND UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1 IS NOT NULL
            AND UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1:eventAction IN('save_changes_to_tree', 'hint_rejected')
            AND domain_sessionid != eligible_events.domain_sessionid
    ) AS has_managed_hint_next_1_days,
    EXISTS (SELECT 1 
        FROM all_events
        WHERE user_id = eligible_events.user_id 
            AND derived_tstamp <= DATEADD(day, 3, eligible_events.derived_tstamp)
            AND UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1 IS NOT NULL
            AND UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1:eventAction IN('save_changes_to_tree', 'hint_rejected')
            AND domain_sessionid != eligible_events.domain_sessionid
    ) AS has_managed_hint_next_3_days,
    EXISTS (SELECT 1 
        FROM all_events
        WHERE user_id = eligible_events.user_id 
            AND derived_tstamp <= DATEADD(day, 7, eligible_events.derived_tstamp)
            AND UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1 IS NOT NULL
            AND UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1:eventAction IN('save_changes_to_tree', 'hint_rejected')
            AND domain_sessionid != eligible_events.domain_sessionid
    ) AS has_managed_hint_next_7_days 
    FROM eligible_events
    GROUP BY ALL
),
aggregated_counts AS (
    SELECT
        user_id,
        derived_tstamp,
        UNSTRUCT_EVENT_COM_FINDMYPAST_HINT_FLOW_1:eventAction AS event_action
    FROM all_events
),
event_time_buckets AS (
    SELECT
        e.event_id,
        e.user_id,
        CASE
            WHEN c.derived_tstamp >= DATEADD(day, -1, e.derived_tstamp) AND c.derived_tstamp <= e.derived_tstamp THEN c.event_action || '_last_1_days'
            WHEN c.derived_tstamp >= DATEADD(day, -3, e.derived_tstamp) AND c.derived_tstamp <= e.derived_tstamp THEN c.event_action || '_last_3_days'
            WHEN c.derived_tstamp >= DATEADD(day, -7, e.derived_tstamp) AND c.derived_tstamp <= e.derived_tstamp THEN c.event_action || '_last_7_days'
            WHEN c.derived_tstamp >= DATEADD(day, -30, e.derived_tstamp) AND c.derived_tstamp <= e.derived_tstamp THEN c.event_action || '_last_30_days'
        END AS feature_name
    FROM eligible_events e
    JOIN aggregated_counts c
      ON c.user_id = e.user_id
     AND c.derived_tstamp <= e.derived_tstamp
),
pivoted_aggregates AS (
    SELECT *
    FROM (
        SELECT
            event_time_buckets.event_id AS pivot_event_id,
            feature_name,
            COUNT(*) AS count
        FROM event_time_buckets
        WHERE feature_name IS NOT NULL
        GROUP BY event_time_buckets.event_id, feature_name
    )
    PIVOT (
        SUM(count)
        FOR feature_name IN (
'drawer_no_hints_last_30_days',
'hint_list_closed_last_30_days',
'view_transcript_from_quick_merge_last_7_days',
'tree_node_hint_click_last_3_days',
'blue_plaque_opened_last_30_days',
'save_changes_to_tree_last_7_days',
'previous_page_last_7_days',
'minimise_explainer_last_30_days',
'finish_last_30_days',
'expand_explainer_last_1_days',
'hints_for_this_tree_last_30_days',
'all_hints_error_last_7_days',
'view_hints_list_for_node_last_30_days',
'drawer_error_last_7_days',
'view_hints_list_for_node_last_1_days',
'drawer_hints_tab_click_last_7_days',
'explore_content_expanded_last_30_days',
'view_source_from_quick_merge_last_3_days',
'hint_rejected_last_30_days',
'all_hints_category_filter_last_30_days',
'drawer_hints_tab_click_last_1_days',
'all_hints_sort_by_last_1_days',
'rejected_hint_error_last_30_days',
'expand_explainer_last_30_days',
'view_hints_list_for_node_last_3_days',
'back_to_tree_last_3_days',
'view_transcript_from_quick_merge_last_30_days',
'hint_card_more_menu_button_click_last_3_days',
'all_hints_no_hints_last_3_days',
'all_hints_no_hints_last_7_days',
'hint_card_more_menu_button_click_last_7_days',
'next_page_last_3_days',
'all_hints_category_filter_last_1_days',
'explore_content_shown_last_7_days',
'back_to_quick_merge_from_transcript_last_7_days',
'minimise_explainer_last_7_days',
'all_hints_category_filter_last_7_days',
'view_transcript_from_quick_merge_last_3_days',
'back_to_tree_last_7_days',
'previous_page_last_1_days',
'full_profile_no_hints_last_3_days',
'explore_content_expanded_last_3_days',
'display_minimised_explainer_last_1_days',
'all_hints_error_last_3_days',
'hint_list_closed_last_1_days',
'previous_page_last_30_days',
'display_expanded_explainer_last_7_days',
'save_changes_to_tree_last_30_days',
'drawer_hints_menu_button_click_last_1_days',
'all_hints_error_last_1_days',
'back_to_quick_merge_from_transcript_last_1_days',
'open_chat_from_quick_merge_last_3_days',
'message_tree_owner_from_quick_merge_last_7_days',
'next_page_last_7_days',
'full_profile_no_hints_last_1_days',
'display_minimised_explainer_last_7_days',
'all_hints_category_filter_no_hints_last_1_days',
'save_changes_to_tree_last_1_days',
'hints_for_this_tree_last_3_days',
'finish_last_1_days',
'open_chat_from_quick_merge_last_30_days',
'all_hints_error_last_30_days',
'minimise_explainer_last_3_days',
'all_hints_clear_filter_last_7_days',
'display_expanded_explainer_last_1_days',
'display_expanded_explainer_last_3_days',
'explore_opened_last_30_days',
'hint_rejected_last_7_days',
'hint_saved_last_7_days',
'rejected_hint_error_last_1_days',
'minimise_explainer_last_1_days',
'drawer_hints_menu_button_click_last_7_days',
'drawer_no_hints_last_1_days',
'expand_explainer_last_3_days',
'finish_last_7_days',
'full_profile_no_hints_last_7_days',
'all_hints_clear_filter_last_30_days',
'explore_content_expanded_last_7_days',
'back_to_quick_merge_from_transcript_last_3_days',
'hint_list_closed_last_3_days',
'all_hints_no_hints_last_1_days',
'back_to_tree_last_1_days',
'hint_opened_last_3_days',
'view_source_from_quick_merge_last_1_days',
'all_hints_no_hints_last_30_days',
'tree_node_hint_click_last_7_days',
'drawer_hints_menu_button_click_last_30_days',
'view_hints_list_for_node_last_7_days',
'drawer_hints_menu_button_click_last_3_days',
'hint_saved_last_30_days',
'hint_card_more_menu_button_click_last_30_days',
'hints_for_this_tree_last_7_days',
'all_hints_sort_by_last_3_days',
'hints_for_this_tree_last_1_days',
'hint_card_more_menu_button_click_last_1_days',
'save_changes_to_tree_last_3_days',
'drawer_no_hints_last_3_days',
'expand_explainer_last_7_days',
'explore_content_shown_last_30_days',
'drawer_error_last_30_days',
'drawer_error_last_3_days',
'previous_page_last_3_days',
'all_hints_clear_filter_last_3_days',
'saved_hint_error_last_7_days',
'drawer_error_last_1_days',
'tree_node_hint_click_last_1_days',
'back_to_quick_merge_from_transcript_last_30_days',
'all_hints_category_filter_last_3_days',
'saved_hint_error_last_1_days',
'message_tree_owner_from_quick_merge_last_30_days',
'hint_opened_last_30_days',
'francis_frith_opened_last_7_days',
'explore_content_shown_last_3_days',
'display_expanded_explainer_last_30_days',
'saved_hint_error_last_3_days',
'all_hints_sort_by_last_7_days',
'all_hints_clear_filter_last_1_days',
'all_hints_sort_by_last_30_days',
'hint_rejected_last_3_days',
'rejected_hint_error_last_3_days',
'back_to_tree_last_30_days',
'finish_last_3_days',
'francis_frith_opened_last_30_days',
'hint_opened_last_1_days',
'explore_content_shown_last_1_days',
'open_chat_from_quick_merge_last_7_days',
'open_chat_from_quick_merge_last_1_days',
'next_page_last_1_days',
'all_hints_category_filter_no_hints_last_7_days',
'message_tree_owner_from_quick_merge_last_3_days',
'message_tree_owner_from_quick_merge_last_1_days',
'view_source_from_quick_merge_last_30_days',
'display_minimised_explainer_last_3_days',
'drawer_no_hints_last_7_days',
'display_minimised_explainer_last_30_days',
'tree_node_hint_click_last_30_days',
'hint_saved_last_1_days',
'drawer_hints_tab_click_last_3_days',
'view_transcript_from_quick_merge_last_1_days',
'all_hints_category_filter_no_hints_last_30_days',
'view_source_from_quick_merge_last_7_days',
'rejected_hint_error_last_7_days',
'hint_rejected_last_1_days',
'hint_list_closed_last_7_days',
'drawer_hints_tab_click_last_30_days',
'full_profile_no_hints_last_30_days',
'next_page_last_30_days',
'hint_saved_last_3_days',
'saved_hint_error_last_30_days',
'hint_opened_last_7_days',
'all_hints_category_filter_no_hints_last_3_days'
        )
    )
),
aggregate_as_of_event AS (
    SELECT
        e.event_id,
        e.user_id,
        -- join in expanded fields once pivoted
        p.*
    FROM eligible_events e
    LEFT JOIN pivoted_aggregates p
      ON e.event_id = p.pivot_event_id
),
final_training AS (
    SELECT
        aggregate_as_of_event.*,
        events_with_target.has_managed_hint_next_1_days,
        events_with_target.has_managed_hint_next_3_days,
        events_with_target.has_managed_hint_next_7_days
    FROM aggregate_as_of_event
    INNER JOIN events_with_target
        ON events_with_target.event_id = aggregate_as_of_event.event_id
)
SELECT
    *,
    CASE WHEN has_managed_hint_next_1_days THEN 1 ELSE 0 END AS label_1,
    CASE WHEN has_managed_hint_next_3_days THEN 1 ELSE 0 END AS label_3,
    CASE WHEN has_managed_hint_next_7_days THEN 1 ELSE 0 END AS label_7
FROM final_training;