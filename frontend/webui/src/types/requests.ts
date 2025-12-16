export interface UserQuery {
    query: string;
}

export interface UserSwitchAgent {
    config_file: string;
}

export interface UserAnswer {
    answer: string;
    ask_id: string;
}

export interface UserRequest {
    type: 'query' | 'list_agents' | 'switch_agent' | 'gen_agent' | 'answer';
    content: UserQuery | UserSwitchAgent | UserAnswer | null;
}